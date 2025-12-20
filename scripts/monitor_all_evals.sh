#!/bin/bash
# Monitor all running evaluations
# Usage: ./scripts/monitor_all_evals.sh

echo "=============================================="
echo "FinSage Evaluation Monitor"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

echo ""
echo "=== 1. AGGRESSIVE (MARFT) - ssh4:36016 ==="
ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -p 36016 root@ssh4.vast.ai "
ps aux | grep evaluate_trained | grep -v grep | head -1 || echo 'Process not running'
echo ''
grep '\[2024-' /root/FinSage/eval_aggressive.log 2>/dev/null | tail -3
" 2>/dev/null || echo "Connection failed"

echo ""
echo "=== 2. FINAGENT FAIR - ssh4:37630 ==="
ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -p 37630 root@ssh4.vast.ai "
ps aux | grep evaluate_baseline | grep -v grep | head -1 || echo 'Process not running'
echo ''
grep -E '(\[2024-|Decision #|Return:)' /root/FinSage/finagent_fair_eval.log 2>/dev/null | tail -5
" 2>/dev/null || echo "Connection failed"

echo ""
echo "=== 3. FINCON FAIR - ssh3:37816 ==="
ssh -o ConnectTimeout=20 -o StrictHostKeyChecking=no -p 37816 root@ssh3.vast.ai "
ps aux | grep evaluate_baseline | grep -v grep | head -1 || echo 'Process not running'
echo ''
grep -E '(\[2024-|Decision #|Return:)' /root/FinSage/fincon_fair_eval.log 2>/dev/null | tail -5
" 2>/dev/null || echo "Connection failed"

echo ""
echo "=============================================="
echo "GPU Costs:"
vastai show instances 2>/dev/null | grep -E "(29036017|29037631|29037816)" | awk '{print $1, $4, $10}'
echo "=============================================="
