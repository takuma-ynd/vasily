Recommendation system (VASILY Internship Assignment)
======

# Usage

--training--
```
python3 main.py --train --train-file ml-100k/u1.base --model-file u1_model.pickle
```

--evaluation--
```
python3 main.py --eval --eval-file ml-100k/u1.test --model-file u1_model.pickle
```

--training & evaluation--
```
python3 main.py --train --train-file ml-100k/u1.base --eval-file ml-100k/u1.test
```

# description
ユーザの各映画への評価からユーザの好みに合った映画をrecommendするシステム．

training時にeval-fileを指定しておけばevaluationも同時に行う．  
model-fileを指定するとモデルの保存・読み込みを行う．  
各ハイパーパラメータも指定できるようになっており，train.shに記したように  
自動でtraining, evaluationを行って結果のログを取らせることも可能．  

モデルは[[Hu et al., 2008]](http://yifanhu.net/PUB/cf.pdf)
を参考にした．


