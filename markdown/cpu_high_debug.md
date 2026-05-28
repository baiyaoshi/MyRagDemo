# CPU 高负载排查指南

## 适用场景
- 服务器 CPU 持续 > 80%
- 应用响应变慢，接口延迟增加
- 用户反馈"系统很卡"
- 告警: `cpu_high_usage` / `load_high`

## 快速定位（30 秒内）

### 1. 看整体负载
```bash
# CPU 使用率和负载
top -bn1 | head -5

# 按 CPU 排序进程
ps aux --sort=-%cpu | head -10

# 每个核心的使用情况
mpstat -P ALL 1 1
```

### 2. 识别 CPU 消耗类型

#### 类型 A: 用户态高 (us > 70%)
通常是业务代码瓶颈：
- 死循环、密集计算
- 正则回溯
- JSON 序列化/反序列化频繁
```python
# Python 中的 CPU 密集型反面示例
while True:
    data = fetch_from_api()  # 没有限流
    process(data)            # 没有 sleep/backpressure
```

#### 类型 B: 内核态高 (sy > 30%)
通常是系统调用频繁：
- 频繁的文件 I/O
- 网络包处理过多
- 锁竞争

#### 类型 C: I/O wait 高 (wa > 20%)
磁盘 I/O 是瓶颈：
- 大量读写日志
- SWAP 频繁换入换出
- 数据库查询全表扫描

#### 类型 D: 软中断高 (si > 10%)
网络包处理瓶颈：
- 网卡队列满
- 频繁的小包网络请求

## Java 应用专项

### 快速定位线程
```bash
# 找到 CPU 最高的 Java 线程
top -H -p <java_pid>

# 将线程 ID 转十六进制
printf "%x\n" <thread_id>

# 查看该线程栈
jstack <java_pid> | grep -A 30 <thread_id_hex>
```

### 常用 JDK 工具
```bash
# CPU 热点采样
profiler.sh -d 30 -f profile.html <java_pid>

# 查看 GC 频率
jstat -gcutil <java_pid> 1000 10
```

## Node.js 应用专项

### CPU 火焰图
```bash
# 使用 perf 生成
perf record -F 99 -p <node_pid> -g --sleep 30
perf script > out.perf

# 使用 0x（需要全局安装）
npx 0x -p <node_pid>
```

### 常见原因
1. 同步阻塞事件循环
2. `JSON.parse` / `JSON.stringify` 大对象
3. 密集的正则匹配
4. 无限递归
5. `while(true)` 没有 `await`

## 通用止损方案

### 紧急
1. 重启服务（临时恢复）
2. 摘除故障节点流量
3. 限流/降级非核心功能

### 长期
1. 加监控：CPU 使用率趋势 + 进程级告警
2. 代码 review：重点排查循环和递归
3. 引入 profiler 定期采样
4. 容量规划：保证有 30% 以上余量

## 参考工具

| 工具 | 用途 | 安装方式 |
|------|------|----------|
| top/htop | 实时进程查看 | 系统自带/apt |
| perf | 性能采样 | linux-tools |
| async-profiler | Java 采样 | GitHub 下载 |
| py-spy | Python 采样 | pip install |
| 0x | Node.js 火焰图 | npm install -g 0x |
