### Table III: Hyperparameter and Circuit Architecture Ablation Study

| configuration                           |   n_qubits |   n_layers | entanglement   |   num_params |   precision@5 |   recall@5 |   ndcg@5 |   hit_rate@5 |   precision@10 |   ndcg@10 |   rmse |
|:----------------------------------------|-----------:|-----------:|:---------------|-------------:|--------------:|-----------:|---------:|-------------:|---------------:|----------:|-------:|
| 2-Qubits, 1-Layer, Entangled            |          2 |          1 | Yes            |            8 |             0 |          0 |        0 |            0 |              0 |         0 | 1.3677 |
| 2-Qubits, 2-Layers, Entangled           |          2 |          2 | Yes            |           12 |             0 |          0 |        0 |            0 |              0 |         0 | 1.7    |
| 4-Qubits, 1-Layer, Entangled            |          4 |          1 | Yes            |           16 |             0 |          0 |        0 |            0 |              0 |         0 | 1.9732 |
| 4-Qubits, 2-Layers, Entangled (Default) |          4 |          2 | Yes            |           24 |             0 |          0 |        0 |            0 |              0 |         0 | 1.5587 |
| 4-Qubits, 3-Layers, Entangled           |          4 |          3 | Yes            |           32 |             0 |          0 |        0 |            0 |              0 |         0 | 1.7496 |
| 4-Qubits, 2-Layers, No Entanglement     |          4 |          2 | No             |           24 |             0 |          0 |        0 |            0 |              0 |         0 | 1.234  |
| 6-Qubits, 2-Layers, Entangled           |          6 |          2 | Yes            |           36 |             0 |          0 |        0 |            0 |              0 |         0 | 1.2159 |