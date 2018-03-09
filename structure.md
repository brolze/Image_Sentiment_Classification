## Structure


| Layer | input | output | activation |
| :------------: |:-------------:| :-----:|:-----:|
| conv2d | [None,48,48,1] | [None,45,45,32] | relu |
| conv2d | [None,45,45,32] | [None,42,42,32] | relu |
| maxpooling | [None,42,42,3] | [None,21,21,32] | valid |
| conv2d | [None,21,21,32] | [None,18,18,32] | relu |
| conv2d | [None,18,18,32] | [None,14,14,32] | relu |
| maxpooling | [None,14,14,32] | [None,7,7,32] | valid |
| flatten | [None,7,7,32] | 1568 | None |
| Dense | 1568 | 128 | sigmoid |
| Dense | 128 | 7 | softmax |


