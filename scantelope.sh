#! /bin/sh

### BEGIN INIT INFO
# Provides:          scantelope
# Required-Start:    $network
# Required-Stop:     $network
# Should-Start:      saned
# Default-Start:     2 3 4 5
# Default-Stop:      1
# Short-Description: Scantelope Scanner Server
# Description:       Scantelope Scanner Server
### END INIT INFO

set -e

DAEMON=/opt/scantelope/serv.py
PID_FILE=/var/run/scantelope.pid

test -x $DAEMON || exit 0

. /lib/lsb/init-functions

case "$1" in
  start)
    log_daemon_msg "Starting scantelope daemon" "scantelope"
    start-stop-daemon --start --quiet --background \
        --pidfile $PID_FILE --make-pidfile \
        --exec $DAEMON
    log_end_msg $?
    ;;

  stop)
    log_daemon_msg "Stopping scantelope daemon" "scantelope"
    start-stop-daemon --stop --quiet --oknodo --pidfile $PID_FILE
    log_end_msg $?
    rm -f $PID_FILE
    ;;

  *)
    echo "Usage: /etc/init.d/scantelope.sh {start|stop}"
    exit 1
esac

exit 0
