f=`ls ~/FCT_scale_v8fromv7`

for x in $f;do
  if ! [ -d "~/FCT_scale_v8fromv7/$x" ]; then
    echo copying $x
    cp ~/FCT_scale_v8fromv7/$x ~/FCT_scale_v11_fromv8/
  fi
done 
