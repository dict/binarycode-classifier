# binarycode-classifier
binary code classifier with deeplearning models(resnet, char-cnn).



### 실험

dataset : 1.6 million binary code of malwares crawled from virus total, 1 million binary code of whitelist(non malware) binary code


| 실험명       | 데이터셋, preprocessing     | 결과   | 설명   |
| --------- | --------------------------- | ---- | ---- |
| Resnet | PE(train-7:eval-3), thumbnail | 85.84% |      |
| Resnet with whitelist | PE(7:3), thumbnail | 63.02% |      |
| distilling nn(model comp.) | PE(7:3), thumbnail | 83.60% | base model : resnet |
| distilling nn(model ensemble) | PE(7:3), thumbnail | 69.50% | base model : resnet |
| MPOS | PE(7:3), 32*24000 | 89.87% |      |
| MPOS with simhash | PE(7:3) 32*24000 | 79.30% | 16bit simhashing with 32 byte block |
| Compare label variance | PE(7:3) | 83% | 각 엔진별 label에 대한 xgboost 결과 |
| apk classification with api call | apk call feature | 96% |      |
| xgboost with whitelist | PE + whitelist 7:3 | 91.8% |      |
| MPOS with whitelist | PE + whitelist 7:3 | 88% |      |