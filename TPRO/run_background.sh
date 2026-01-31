#!/bin/bash
# Chạy train khi tắt máy / disconnect vẫn chạy.
# ./run_background.sh nohup work_dirs/bcss/classification/config.yaml
# ./run_background.sh tmux work_dirs/bcss/classification/config.yaml

set -e
CONFIG="${2:-work_dirs/bcss/classification/config.yaml}"
cd "$(dirname "$0")"

case "${1:-tmux}" in
  nohup)
    nohup python -u train_cls.py --config "$CONFIG" > "nohup_$(basename "$CONFIG" .yaml).out" 2>&1 &
    echo "PID: $!"
    ;;
  tmux)
    SESSION="tpro_$(basename "$CONFIG" .yaml)"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
      echo "Session '$SESSION' đã tồn tại. Gõ: tmux attach -t $SESSION"
      exit 1
    fi
    tmux new-session -d -s "$SESSION" "python -u train_cls.py --config '$CONFIG'; echo 'Done.'; read"
    echo "Chạy trong tmux: $SESSION. Xem: tmux attach -t $SESSION. Thoát: Ctrl+B d"
    ;;
  *)
    echo "Usage: $0 {nohup|tmux} [config.yaml]"
    exit 1
    ;;
esac
