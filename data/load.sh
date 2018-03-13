#!/bin/bash

project=$1;

root="/home/egor/PycharmProjects/stan/data"
reps="$root/$project/reps";
repState="$root/$project/repositoryState";
files="$repState/reps_to_load.csv";

cat "$files" | while read line
do
    if [[ ${line} == /* ]]; then
        continue;
    fi
    cd ${reps};
    stage="address";
    address="";
    version="";
    for word in ${line}; do
        if [ ${stage} = "address" ]; then
            address=${word};
            stage="version";
        else
            version=${word};
        fi
    done
    echo ${address};
    echo ${version};
    name="$address";
    name="${name##*/}";
    name="${name%.*}";
    error=$(timeout 60 git clone ${address} 2>&1 );
    if echo "$error" | grep -q "fatal" ; then
        rm -rf "$name";
        continue;
    fi
    error=$( cd "${reps}/${name}" 2>&1 );
    if echo "$error" | grep -q "No such file or directory" ; then
        continue;
    fi
    cd "${reps}/${name}"
    git checkout ${version};
    author="";
    failed="False";
    while read check_author
    do
        if [ "$author" = "" ]; then
            author=${check_author};
        elif [ "$author" != "$check_author" ]; then
            failed="True";
        fi
    done < <(git log | grep "Author");
    cd ${reps};
    if [ "$failed" = "True" ]; then
        rm -rf "${reps}/${name}";
    else
        echo "$address $version" >> "$repState/clean_reps.csv"
        cd "${reps}/${name}"
        for longName in $(find | grep "\.java$")
        do
            shortName="${longName##*/}"
            mv "${longName}" "./${shortName}"
        done
        find . ! -regex '.*\.java' -type f -exec rm -f {} +
        find . ! -regex '\.' -type d -exec rm -rd {} +
    fi

done
