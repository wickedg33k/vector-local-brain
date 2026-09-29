#!/bin/bash
# Vector stability probe (Hermes, 2026-09-26). One line per run -> health.log
ts=$(date "+%F %T")
loss=$(ping -c 10 -i 0.3 -W 1 ROBOT_IP 2>/dev/null | grep -o "[0-9.]*% packet loss" | cut -d% -f1)
rtt=$(ping -c 3 -i 0.3 -W 1 ROBOT_IP 2>/dev/null | awk -F/ "/rtt/{print \$5}")
pod=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:8080/api/get_config)
brain=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://127.0.0.1:11500/v1/models)
reqs=$(docker logs --since 5m vector-pod 2>&1 | grep -c "Transcribed text")
state=$(docker inspect -f "{{.State.Status}}/{{.RestartCount}}" vector-pod 2>/dev/null)
echo "$ts loss=${loss:-100}% rtt=${rtt:-na}ms pod=$pod brain=$brain voice_reqs_5m=$reqs container=$state" >> ~/vector-health/health.log
