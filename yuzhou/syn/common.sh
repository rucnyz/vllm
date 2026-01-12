#!/bin/bash
# 公共函数库 - 被其他实验脚本 source 使用
#
# 用法: source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#
# 提供的函数:
#   setup_cleanup BASE_PORT      - 设置 Ctrl+C 清理 (需要先定义 WORKER_PIDS 数组)
#   detect_available_gpus        - 检测可用 GPU，返回空格分隔的 GPU ID 列表
#   check_port_available PORT    - 检查端口是否可用，不可用则打印错误并返回 1
#   wait_for_server PORT PID [MAX_WAIT] - 等待服务启动，默认最多等待 180 秒
#   kill_server PID              - 终止服务进程
#
# 提供的变量:
#   GPU_MEM_THRESHOLD - GPU 内存阈值 (MiB)，默认 10000

# 默认配置
GPU_MEM_THRESHOLD=${GPU_MEM_THRESHOLD:-10000}

# ========================================
# 清理函数
# ========================================

# 设置清理函数和 trap
# 参数: $1 = BASE_PORT
setup_cleanup() {
    local base_port=${1:-8200}

    # 定义清理函数
    _cleanup() {
        echo ""
        echo "收到中断信号，正在清理..."

        # 终止所有 worker 进程
        if [ ${#WORKER_PIDS[@]} -gt 0 ]; then
            for pid in "${WORKER_PIDS[@]}"; do
                kill -9 $pid 2>/dev/null || true
            done
        fi

        # 终止所有使用 BASE_PORT 范围的 vllm 进程
        for ((i=0; i<10; i++)); do
            local port=$(($base_port + i))
            pkill -9 -f "vllm.*--port $port" 2>/dev/null || true
        done

        echo "清理完成"
        exit 1
    }

    # 捕获 Ctrl+C (SIGINT) 和 SIGTERM
    trap _cleanup SIGINT SIGTERM
}

# ========================================
# GPU 检测
# ========================================

# 检测可用 GPU (内存使用低于阈值的)
# 输出: 空格分隔的 GPU ID 列表
detect_available_gpus() {
    local available=()

    while IFS=, read -r gpu_id name mem_total mem_used; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        mem_used=$(echo "$mem_used" | tr -d ' MiB')

        if [ "$mem_used" -lt "$GPU_MEM_THRESHOLD" ]; then
            available+=("$gpu_id")
        fi
    done < <(nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader 2>/dev/null)

    echo "${available[@]}"
}

# 选择要使用的 GPU
# 参数: $1 = MAX_GPUS (最大使用数量)
# 输出: 设置 GPUS_TO_USE 和 NUM_GPUS 变量
select_gpus() {
    local max_gpus=${1:-4}

    AVAILABLE_GPUS=($(detect_available_gpus))
    NUM_AVAILABLE=${#AVAILABLE_GPUS[@]}

    if [ "$NUM_AVAILABLE" -eq 0 ]; then
        echo "错误: 没有可用的 GPU (内存使用 < ${GPU_MEM_THRESHOLD} MiB)"
        exit 1
    fi

    if [ "$NUM_AVAILABLE" -gt "$max_gpus" ]; then
        GPUS_TO_USE=("${AVAILABLE_GPUS[@]:0:$max_gpus}")
    else
        GPUS_TO_USE=("${AVAILABLE_GPUS[@]}")
    fi

    NUM_GPUS=${#GPUS_TO_USE[@]}

    echo "检测到 $NUM_AVAILABLE 张可用 GPU: ${AVAILABLE_GPUS[*]}"
    echo "将使用 $NUM_GPUS 张 GPU: ${GPUS_TO_USE[*]}"
}

# ========================================
# 端口检测
# ========================================

# 检查端口是否可用
# 参数: $1 = PORT, $2 = GPU_ID (可选，用于日志)
# 返回: 0 = 可用, 1 = 被占用
check_port_available() {
    local port=$1
    local gpu_id=${2:-""}
    local prefix=""

    if [ -n "$gpu_id" ]; then
        prefix="[GPU $gpu_id] "
    fi

    if lsof -i :$port >/dev/null 2>&1; then
        echo "${prefix}错误: 端口 $port 已被占用!"
        echo "${prefix}占用进程:"
        lsof -i :$port 2>/dev/null | head -5
        echo "${prefix}请先终止占用进程: pkill -9 -f 'vllm.*--port $port'"
        return 1
    fi
    return 0
}

# ========================================
# 服务管理
# ========================================

# 等待服务启动
# 参数: $1 = PORT, $2 = SERVER_PID, $3 = MAX_WAIT (可选，默认 180)
# 返回: 0 = 成功, 1 = 超时或进程退出
wait_for_server() {
    local port=$1
    local server_pid=$2
    local max_wait=${3:-180}
    local wait_count=0

    while [ $wait_count -lt $max_wait ]; do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            return 0
        fi
        # 检查进程是否还存活
        if ! kill -0 $server_pid 2>/dev/null; then
            echo "服务进程意外退出"
            return 1
        fi
        sleep 1
        wait_count=$((wait_count + 1))
    done

    echo "服务启动超时 (${max_wait}s)"
    return 1
}

# 终止服务进程
# 参数: $1 = SERVER_PID
kill_server() {
    local server_pid=$1
    kill -INT $server_pid 2>/dev/null || true
    sleep 2
    kill -9 $server_pid 2>/dev/null || true
}

# ========================================
# 并行调度器
# ========================================

# 从队列获取下一个实验 (带锁)
# 参数: $1 = QUEUE_FILE, $2 = LOCK_FILE
# 输出: 实验配置字符串，或空
get_next_experiment() {
    local queue_file=$1
    local lock_file=$2

    (
        flock -x 200
        local exp=$(head -n 1 "$queue_file" 2>/dev/null)
        if [ -n "$exp" ]; then
            tail -n +2 "$queue_file" > "${queue_file}.tmp"
            mv "${queue_file}.tmp" "$queue_file"
            echo "$exp"
        fi
    ) 200>"$lock_file"
}

# 更新进度
# 参数: $1 = STATUS, $2 = PROGRESS_FILE, $3 = LOCK_FILE, $4 = TOTAL_EXPERIMENTS
update_progress() {
    local status=$1
    local progress_file=$2
    local lock_file=$3
    local total=$4

    (
        flock -x 200
        echo "$status" >> "$progress_file"
        local completed=$(wc -l < "$progress_file")
        local remaining=$((total - completed))
        echo "进度: $completed / $total (剩余 $remaining)"
    ) 200>"$lock_file"
}

# ========================================
# 环境设置
# ========================================

# 初始化实验环境
# 参数: $1 = VENV_PATH (可选)
init_experiment_env() {
    local venv_path=${1:-"/scratch/yuzhou/aproj/vllm/.venv"}

    # 增加文件描述符限制
    ulimit -n 65535 2>/dev/null || true

    # 激活虚拟环境
    if [ -f "${venv_path}/bin/activate" ]; then
        source "${venv_path}/bin/activate"
    fi
}

# ========================================
# 结果统计
# ========================================

# 打印实验结果统计
# 参数: $1 = PROGRESS_FILE, $2 = TOTAL_EXPERIMENTS, $3 = OUTPUT_DIR
print_summary() {
    local progress_file=$1
    local total=$2
    local output_dir=$3

    local completed=$(wc -l < "$progress_file" 2>/dev/null || echo 0)
    local ok=$(grep -c "^OK|" "$progress_file" 2>/dev/null || echo 0)
    local fail=$(grep -c "^FAIL|" "$progress_file" 2>/dev/null || echo 0)

    echo ""
    echo "========================================"
    echo "所有实验完成！"
    echo "========================================"
    echo "总计: $completed / $total"
    echo "成功: $ok"
    echo "失败: $fail"
    echo ""
    echo "结果目录: $output_dir"
}
