#!/bin/bash
# ROCm + vLLM + ModelScope 一键管理脚本（支持容器内下载&服务启动）
# 使用前：确保有一个名为 rocm 的镜像（或改成你的镜像名），其中已包含 vLLM。
# 注意：本脚本改为使用 ModelScope 下载模型，不再依赖 HF_TOKEN。

# ======= 配置区 =======
CONTAINER_NAME="rocm"
# ModelScope 的模型ID（你要的 3.1 8B Instruct）
MS_MODEL_ID="LLM-Research/Meta-Llama-3.1-8B-Instruct"

# 模型本地落地目录（挂载卷，容器内外一致）
MODEL_DIR_HOST="/shared-docker/models/Meta-Llama-3.1-8B-Instruct"
MODEL_DIR_IN_CONTAINER="$MODEL_DIR_HOST"   # 因为挂了同一路径，容器内就是这个绝对路径

# vLLM served 名称 & 端口
SERVED_MODEL_NAME="llama-3.1-8b-instruct"
PORT="8001"

# 日志
LOG_DIR="/root/vllm_logs"
PID_FILE="$LOG_DIR/vllm_docker.pid"

# 你现有的 ROCm/vLLM 基础镜像名（根据实际情况修改）
IMAGE_NAME="rocm"

# 额外挂载（HF 缓存可选，ModelScope 我们用自定义目录）
HF_CACHE_HOST="$HOME/.cache/huggingface"
MODELSCOPE_CACHE_HOST="/shared-docker/modelscope_cache"

# vLLM 运行参数（按需调整）
MAX_MODEL_LEN="131072"               # Llama3.1 支持 128K，上下取整给 131072
GPU_MEM_UTIL="0.95"

mkdir -p "$LOG_DIR" "$MODEL_DIR_HOST" "$MODELSCOPE_CACHE_HOST" "$HF_CACHE_HOST"

# ======= 函数区 =======

recreate_container() {
    echo "🔧 重新创建容器（端口映射模式）..."
    docker exec $CONTAINER_NAME pkill -f "vllm serve" 2>/dev/null || true
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true

    mkdir -p /shared-docker
    mkdir -p "$HF_CACHE_HOST"

    docker run -it -d \
      --name $CONTAINER_NAME \
      --device=/dev/kfd \
      --device=/dev/dri \
      --group-add=video \
      --ipc=host \
      --cap-add=SYS_PTRACE \
      --security-opt seccomp=unconfined \
      -p 8000:8000 -p 8888:8888 -p 30000:30000 -p 8001:8001 \
      -v /shared-docker:/shared-docker \
      -v "$HF_CACHE_HOST":/root/.cache/huggingface \
      -v "$MODELSCOPE_CACHE_HOST":/root/.cache/modelscope \
      "$IMAGE_NAME"

    if [ $? -eq 0 ]; then
        echo "✅ 容器重新创建成功"
        sleep 5
        return 0
    else
        echo "❌ 容器创建失败"
        return 1
    fi
}

ensure_port_mapping() {
  # 检查 rocm 是否已发布 8001
  if ! docker ps --format '{{.Names}}\t{{.Ports}}' \
      | grep -qE "^$CONTAINER_NAME\s+.*8001->8001"; then
    echo "⚠️  检测到容器 $CONTAINER_NAME 未发布端口 8001，自动重建容器..."
    recreate_container || {
      echo "❌ 自动重建容器失败，请手动执行：$0 container recreate"; exit 1;
    }
  fi
}


ensure_modelscope_and_download() {
    echo "📦 确认容器内已安装 ModelScope，并下载模型到持久卷..."
    if ! docker ps | grep -q $CONTAINER_NAME; then
        echo "❌ 容器 $CONTAINER_NAME 未运行，请先: $0 container recreate 或 $0 container start"
        exit 1
    fi

    # 容器内安装 modelscope（若已安装会跳过）
    docker exec $CONTAINER_NAME bash -lc "
        python -c 'import modelscope' 2>/dev/null || pip install -U modelscope
    " || {
        echo "❌ 安装 modelscope 失败"; exit 1;
    }

    # 下载（使用 --local_dir 把权重直接落到挂载卷里）
    if docker exec $CONTAINER_NAME test -f \"$MODEL_DIR_IN_CONTAINER/config.json\"; then
        echo "✅ 已检测到本地模型: $MODEL_DIR_IN_CONTAINER"
    else
        echo "⬇️  正在下载 $MS_MODEL_ID 到 $MODEL_DIR_IN_CONTAINER ..."
        docker exec $CONTAINER_NAME bash -lc "
            mkdir -p \"$MODEL_DIR_IN_CONTAINER\"
            modelscope download --model \"$MS_MODEL_ID\" --local_dir \"$MODEL_DIR_IN_CONTAINER\"
        " || { echo "❌ ModelScope 下载失败"; exit 1; }
        echo "✅ 下载完成"
    fi
}

start_vllm() {
    echo "🚀 启动 vLLM（ModelScope 本地模型）..."
    if ! docker ps | grep -q $CONTAINER_NAME; then
        echo "❌ 容器未运行"; return 1; fi
    ensure_port_mapping
    
    if docker exec $CONTAINER_NAME pgrep -f "vllm serve" >/dev/null; then
        echo "❌ vLLM 已在运行"; return 1; fi

    # 先保证模型可用
    ensure_modelscope_and_download

    # 启动 vLLM（用本地目录作为 --model）
    nohup docker exec $CONTAINER_NAME bash -lc "
        mkdir -p /app/logs
        cd /app
        nohup vllm serve \"$MODEL_DIR_IN_CONTAINER\" \
            --host 0.0.0.0 \
            --port $PORT \
            --served-model-name \"$SERVED_MODEL_NAME\" \
            --max-model-len $MAX_MODEL_LEN \
            --gpu-memory-utilization $GPU_MEM_UTIL \
            --enable-chunked-prefill \
            > /app/logs/vllm.log 2>&1 & echo \$! > /app/vllm.pid
        echo 'vLLM 服务已在容器内启动'
    " > "$LOG_DIR/docker_exec.log" 2>&1 &

    echo $! > "$PID_FILE"
    echo "✅ 启动命令已发送，等待服务就绪..."
    sleep 15
    check_service
}

stop_vllm() {
    echo "🛑 停止 vLLM..."
    docker exec $CONTAINER_NAME bash -lc '
        if [ -f /app/vllm.pid ]; then
            PID=$(cat /app/vllm.pid)
            if kill -0 $PID 2>/dev/null; then
                kill -TERM $PID; sleep 5
                kill -0 $PID 2>/dev/null && kill -KILL $PID
            fi
            rm -f /app/vllm.pid
        else
            pkill -f "vllm serve" || true
        fi
        echo "vLLM 已停止"
    '
    rm -f "$PID_FILE"
    echo "✅ vLLM 服务已停止"
}

restart_vllm(){ stop_vllm; sleep 3; start_vllm; }

check_service() {
    echo "🔍 检查服务状态..."
    docker ps | grep -q $CONTAINER_NAME && echo "✅ 容器运行中" || { echo "❌ 容器未运行"; return 1; }

    if docker exec $CONTAINER_NAME pgrep -f "vllm serve" >/dev/null; then
        PID=$(docker exec $CONTAINER_NAME pgrep -f "vllm serve")
        echo "✅ vLLM 进程运行中 (PID: $PID)"
    else
        echo "❌ vLLM 进程未运行"; return 1;
    fi

    echo "🌐 容器内 API 检查..."
    docker exec $CONTAINER_NAME curl -s --connect-timeout 5 "http://localhost:$PORT/v1/models" >/dev/null && \
        echo "✅ 容器内 API 正常" || { echo "❌ 容器内 API 异常"; return 1; }

    echo "🌐 宿主机 API 检查..."
    curl -s --connect-timeout 5 "http://localhost:$PORT/v1/models" >/dev/null && \
        echo "✅ 宿主机 API 正常；测试：curl http://localhost:$PORT/v1/models" || {
        echo "❌ 宿主机 API 无响应，建议：$0 container recreate"; return 1; }
}

show_logs() {
    echo "📝 vLLM 日志（容器内最近 50 行）:"
    docker exec $CONTAINER_NAME bash -lc 'test -f /app/logs/vllm.log && tail -50 /app/logs/vllm.log || echo "无日志"'
    echo -e "\n📝 宿主机执行日志:"
    test -f "$LOG_DIR/docker_exec.log" && tail -100 "$LOG_DIR/docker_exec.log" || echo "无宿主机日志"
}

follow_logs(){ docker exec -it $CONTAINER_NAME bash -lc 'tail -f /app/logs/vllm.log'; }

test_service() {
    echo "🧪 测试 /v1/chat/completions..."
    start_time=$(date +%s.%3N)
    resp=$(curl -s -X POST "http://localhost:$PORT/v1/chat/completions" \
      -H "Content-Type: application/json" -d "{
        \"model\": \"$SERVED_MODEL_NAME\",
        \"messages\": [{\"role\":\"user\",\"content\":\"Say: Hi there!\"}],
        \"max_tokens\": 8
      }")
    end_time=$(date +%s.%3N)
    if echo "$resp" | grep -q '"choices"'; then
        echo "✅ 成功，耗时 $(echo \"$end_time - $start_time\" | bc 2>/dev/null)s"
        echo "🗨️  回复：$(echo "$resp" | jq -r '.choices[0].message.content' 2>/dev/null)"
    else
        echo "❌ 失败，原始响应：$resp"
    fi
}

manage_container() {
    case "$1" in
      start)   docker start $CONTAINER_NAME ;;
      stop)    docker stop $CONTAINER_NAME ;;
      restart) docker restart $CONTAINER_NAME ;;
      recreate) recreate_container ;;
      *) echo "用法: $0 container {start|stop|restart|recreate}";;
    esac
}

show_help() {
    cat <<EOF
🐳 ROCm vLLM + ModelScope 管理脚本

用法: $0 {start|stop|restart|status|logs|follow|test|container}

命令说明：
  start     - 安装/下载(如需)并启动 vLLM（使用 ModelScope 本地模型）
  stop      - 停止 vLLM
  restart   - 重启 vLLM
  status    - 检查服务状态
  logs      - 查看日志
  follow    - 实时跟随日志
  test      - 发送测试请求
  container start|stop|restart|recreate - 容器管理（recreate 会挂载缓存并映射端口）

当前配置：
  容器名: $CONTAINER_NAME
  镜像:   $IMAGE_NAME
  模型ID: $MS_MODEL_ID
  模型目录(宿主/容器): $MODEL_DIR_HOST
  端口:   $PORT
  日志:   $LOG_DIR
EOF
}

case "$1" in
  start)    start_vllm ;;
  stop)     stop_vllm ;;
  restart)  restart_vllm ;;
  status)   check_service ;;
  logs)     show_logs ;;
  follow)   follow_logs ;;
  test)     test_service ;;
  container) manage_container "$2" ;;
  help|--help|-h|"") show_help ;;
  *) echo "❌ 未知命令: $1"; show_help; exit 1 ;;
esac
