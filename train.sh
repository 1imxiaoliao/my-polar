model=w_u2net
python train.py \
  --batch-size 16 \
  --epochs 200 \
  --logs ./logs/$model \
  --model $model \
  --dataset liver \
  --percent 1.0