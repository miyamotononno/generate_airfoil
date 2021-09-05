# generate_airfoil

## 本リポジトリについて
本リポジトリは東京大学工学部構造研究室で行っていた卒業研究のソースコードです。 
conditional GANを用いた翼型の形状生成を行いました。
詳細は、こちらの[卒業研究発表用スライド](https://github.com/miyamotononno/generate_airfoil/issues/13) を参考にしてください。  
さらに詳しい内容を知りたい方は、 nozomiya.422@gmail.com までお問い合わせください。

追記(2021/09/05)：
Google Scholor上で論文を閲覧できるようになりました。  
題名: Inverse Airfoil Design Method for Generating　Varieties of Smooth Airfoils Using Conditional　WGAN-GP  
リンク: https://assets.researchsquare.com/files/rs-618399/v1_covered.pdf?c=1624375613

## setup
本研究では、深層学習フレームワークとして[pytorch](https://pytorch.org/)を使用しているので、それのインストールが必要となります。  
さらに、形状から揚力計算を行うcalc_cl.pyファイルを走らせる際は、XFOILを使用しているので、それの[python用ライブラリ](https://github.com/KikeM/xfoil-python)をインストールする必要があります。  
しかし、そのインストールは少々難しいので、[こちら](https://github.com/miyamotononno/generate_airfoil/issues/14)を参考にしてください。

## run

代表的なコマンドを以下に記します。
```
通常のconditional GANの訓練
python3 -m normal.train

通常のconditional GANの評価
python3 -m normal.eval

conditional wgan-gpの訓練
python3 -m wgan_gp.train

conditional wgan-gpの評価
python3 -m wgan_gp.eval
```

その他にも様々な関数があります。ご自身で、適宜変更してください。
