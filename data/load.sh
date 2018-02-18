#!/bin/bash

project=$1;

root="/home/egor/PyCharm2017/projects/stan/data"
files="$root/$project/repositoryState/reps.csv";
reps="$root/$project/reps";
repState="$root/$project/repositoryState";

cat "$files" | while read line
do
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
    error=$( git clone ${address} 2>&1 );
    if echo "$error" | grep -q "fatal" ; then
        rm -rf "$name";
        continue;
    fi
    error=$( cd "${reps}/${name}" 2>&1 );
    if echo "$error" | grep -q "No such file or directory" ; then
        continue;
    fi
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
        rm -rf "$name";
    else
        echo "$address $version" >> "$repState/clean_reps.csv"
    fi

done
