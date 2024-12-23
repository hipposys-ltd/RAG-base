#!/usr/bin/env bash

# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Get the directory this script is located in (regardless of where it is called from)
MILVUS_DIR=$(cd `dirname $0`; pwd)

# Make sure that the YAML files exist. This is to avoid a bug that causes these files to be created as directories.
touch ${MILVUS_DIR}/embedEtcd.yaml
touch ${MILVUS_DIR}/user.yaml

run_embed() {
    cat << EOF > embedEtcd.yaml
listen-client-urls: http://0.0.0.0:8379
advertise-client-urls: http://0.0.0.0:8379
quota-backend-bytes: 4294967296
auto-compaction-mode: revision
auto-compaction-retention: '1000'
EOF

    cat << EOF > user.yaml
# Extra config to override default milvus.yaml
EOF

    docker run -d \
        --name milvus-standalone \
        --security-opt seccomp:unconfined \
        -e ETCD_USE_EMBED=true \
        -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
        -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
        -e COMMON_STORAGETYPE=local \
        -v ${MILVUS_DIR}/volumes/milvus:/var/lib/milvus \
        -v ${MILVUS_DIR}/embedEtcd.yaml:/milvus/configs/embedEtcd.yaml \
        -v ${MILVUS_DIR}/user.yaml:/milvus/configs/user.yaml \
        -p 19530:19530 \
        -p 9091:9091 \
        -p 8379:2379 \
        --health-cmd="curl -f http://localhost:9091/healthz" \
        --health-interval=30s \
        --health-start-period=90s \
        --health-timeout=20s \
        --health-retries=3 \
        milvusdb/milvus:v2.4.5 \
        milvus run standalone  1> /dev/null
}

wait_for_milvus_running() {
    echo "Wait for Milvus Starting..."
    while true
    do
        res=`docker ps|grep milvus-standalone|grep healthy|wc -l`
        if [ $res -eq 1 ]
        then
            echo "Start successfully."
            echo "To change the default Milvus configuration, add your settings to the user.yaml file and then restart the service."
            break
        fi
        sleep 1
    done
}

start() {
    res=`docker ps|grep milvus-standalone|grep healthy|wc -l`
    if [ $res -eq 1 ]
    then
        echo "Milvus is running."
        exit 0
    fi

    res=`docker ps -a|grep milvus-standalone|wc -l`
    if [ $res -eq 1 ]
    then
        docker start milvus-standalone 1> /dev/null
    else
        run_embed
    fi

    if [ $? -ne 0 ]
    then
        echo "Start failed."
        exit 1
    fi

    wait_for_milvus_running
}

stop() {
    docker stop milvus-standalone 1> /dev/null

    if [ $? -ne 0 ]
    then
        echo "Stop failed."
        exit 1
    fi
    echo "Stop successfully."

}

delete() {
    res=`docker ps|grep milvus-standalone|wc -l`
    if [ $res -eq 1 ]
    then
        echo "Please stop Milvus service before delete."
        exit 1
    fi
    docker rm milvus-standalone 1> /dev/null
    if [ $? -ne 0 
    then
        echo "Delete failed."
        exit 1
    fi
    rm -rf ${MILVUS_DIR}/volumes
    rm -rf ${MILVUS_DIR}/embedEtcd.yaml
    rm -rf ${MILVUS_DIR}/user.yaml
    echo "Delete successfully."
}


case $1 in
    restart)
        stop
        start
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    delete)
        delete
        ;;
    *)
        echo "please use bash standalone_embed.sh restart|start|stop|delete"
        ;;
esac
