# tensorflow 集群安装文件：
1. check.sh 可以放在tensorflow集群的任意机器，只需要一台机器，ip 为配置文件server-config.xml中的 tensorflow master_ip。其中check.sh中涉及到用户，默认为root。需要确保server-config中配置的tensorflow user用户和check.sh中的用户能够免密访问，且不出现任何警告。
2. 每台配置有 tensorflow 的机器都需要 check_port.sh ,放在配置的tensorflow user账号的根目录下。配置账号需要 sudo 权限。
3. 每台配置有 tensorflow 的机器都需要有 /usr/bda/port 目录；
4. 每台配置有 tensorflow的机器都需要有/var/bda_tensorflow 目录
5. 每台配置有 tensorflow 的机器都需要 start.py  start.sh，用于模型并行，放在配置的放在配置的tensorflow user账号的根目录下。
6. 保证tensorflow集群中的任意机器两两通过tensorflow user帐号能够免密登入且不产生警告
