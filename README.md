# generate_airfoil

## 本リポジトリについて
本リポジトリは東京大学工学部構造研究室で行っていた卒業研究のソースコードです。 
conditional GANを用いた翼型の形状生成を行いました。
詳細は、こちらの[卒業研究発表用スライド](https://github.com/miyamotononno/generate_airfoil/issues/13) を参考にしてください。  
さらに詳しい内容を知りたい方は、 nozomiya.422@gmail.com までお問い合わせください。

## setup
本研究では、深層学習フレームワークとして[pytorch](https://pytorch.org/)を使用しているので、それのインストールが必要となります。  
さらに、形状から揚力計算を行うcalc_cl.pyファイルを走らせる際は、XFOILを使用しているので、それの[python用ライブラリ](https://github.com/KikeM/xfoil-python)をインストールする必要があります。  
しかし、そのインストールは少々難しいので、[こちら](https://github.com/miyamotononno/generate_airfoil/issues/14)を参考にしてください。

## run
