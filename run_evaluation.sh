for f in output/*.json ; 

do
echo Processing file: $f
f_eval=${f/output/results}
f_eval=${f_eval/json/eval.json}

echo Saving to: $f_eval

python evaluation.py  \
--prediction_file $f \
--output_path $f_eval ;

echo Saved to: $f_eval
echo ================================

done
