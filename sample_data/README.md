### モデルについて
[tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3) で作成したモデルを前提としています。

学習済みモデルのグラフデータを yolov3_gpu_nms.pb というファイル名で実行可能ファイルと同じディレクトリに置いてください。

~~自身で学習したモデルデータの場合、~~ *.weightファイルの場合は convert_weight.py で変換して出力された上記ファイルを使用してください。

とりあえず実行してみたい場合は、 [tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3#part-2-quick-start) のyolov3.weightを使用してみてください。

### ラベルについて
実行可能ファイルと同じディレクトリに labels.txt（UTF-16）を作成し、ラベル名を改行を区切りとして記述してください。

プログラム内では行番号をラベルIDとして扱います。  
e.g.  
`label_map.pbtxt`などが次の場合、
```
{
    id: 1
    name: "person"
}
{
    id: 2
    name: "bicycle"
}
```
`labels.txt`は次の通り。
```
person
bicycle
```

[tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3#part-2-quick-start) のモデルデータのラベルは /sample_data/labels.txt に置いてあります（たぶんこれで動くはず）。