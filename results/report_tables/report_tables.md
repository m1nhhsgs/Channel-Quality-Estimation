# Report Tables - ESP-NOW Packet Loss Prediction


Current model: XGBoost regressor without session metadata. Target: `packet_loss_future_2s`. State classes: `Good` and `Critical`.


## 1. Dataset Overview


| item | value |
| --- | --- |
| Raw sessions/runs in feature dataset | 19 |
| Sliding-window feature rows | 11511 |
| Feature columns used by current model | 15 |
| Window size | 2 seconds |
| Step size | 0.5 seconds |
| Prediction horizon | 2 seconds |
| Target | packet_loss_future_2s |
| Train/test split | group_by_scenario_id |
| Train rows | 8583 |
| Test rows | 2928 |
| State classes | Good, Critical |
| Critical threshold | packet_loss_future_2s >= 10% |

## 2. Model Metrics


| metric | value | unit | note |
| --- | --- | --- | --- |
| MAE | 8.3173 | percentage points | Lower is better |
| RMSE | 13.116 | percentage points | Lower is better |
| R2 | -0.6171 |  | Higher is better |
| Binary state accuracy | 0.779 | ratio | Good/Critical accuracy |
| Binary F1 macro | 0.7605 | ratio | Average F1 over Good and Critical |

## 3. State Thresholds


| state | packet_loss_future_2s |
| --- | --- |
| Good | < 10% |
| Critical | >= 10% |

## 4. Confusion Matrix


| actual_state | Good | Critical |
| --- | --- | --- |
| Good | 1548 | 633 |
| Critical | 14 | 733 |

## 5. Full Dataset State Distribution


| state | rows | percent |
| --- | --- | --- |
| Good | 6776 | 58.87 |
| Critical | 4735 | 41.13 |

## 6. Test State Distribution


| state | rows | percent |
| --- | --- | --- |
| Good | 2181 | 74.49 |
| Critical | 747 | 25.51 |

## 7. Test Performance By Run


| scenario_id | rows | actual_loss_mean_pct | predicted_loss_mean_pct | mae | rmse | binary_state_accuracy | actual_good_rows | predicted_good_rows | actual_critical_rows | predicted_critical_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run02 | 592 | 0.0 | 1.967 | 1.967 | 2.327 | 1.0 | 592 | 592 | 0 | 0 |
| run03 | 593 | 0.034 | 1.92 | 1.913 | 2.274 | 1.0 | 593 | 593 | 0 | 0 |
| run14 | 980 | 0.0 | 16.925 | 16.925 | 21.097 | 0.3694 | 980 | 362 | 0 | 618 |
| run50 | 763 | 22.456 | 20.519 | 7.166 | 8.959 | 0.962 | 16 | 15 | 747 | 748 |

## 8. Top 15 Feature Importance


| feature | importance |
| --- | --- |
| packet_loss_past_2s | 0.496651 |
| jitter_2s | 0.087741 |
| rssi_min_2s | 0.078168 |
| throughput_past_2s | 0.071714 |
| rssi_mean_2s | 0.058755 |
| packet_rate | 0.035203 |
| entropy_rssi_2s | 0.033178 |
| inter_arrival_mean_2s | 0.032914 |
| rssi_max_2s | 0.0271 |
| rssi_slope_2s | 0.018542 |
| rssi_std_2s | 0.017741 |
| rssi_range_2s | 0.016769 |
| rssi_last | 0.015808 |
| rssi_delta_2s | 0.009717 |
| payload_size | 0.0 |

## 9. Binary Classification Details


| metric | value | note |
| --- | --- | --- |
| Accuracy | 0.779 | Overall Good/Critical classification accuracy |
| F1 macro | 0.7605 | Average F1 over Good and Critical |
| Critical precision | 0.5366 | Among predicted Critical windows, fraction that are truly Critical |
| Critical recall | 0.9813 | Among actual Critical windows, fraction detected as Critical |

## 10. Feature Dataset By Run


| scenario_id | feature_rows | loss_mean_pct | loss_min_pct | loss_max_pct | good_rows | critical_rows |
| --- | --- | --- | --- | --- | --- | --- |
| run02 | 592 | 0.0 | 0.0 | 0.0 | 592 | 0 |
| run03 | 593 | 0.034 | 0.0 | 2.5 | 593 | 0 |
| run04 | 606 | 2.011 | 0.0 | 59.524 | 561 | 45 |
| run05 | 1026 | 4.465 | 0.0 | 94.444 | 928 | 98 |
| run13 | 594 | 0.0 | 0.0 | 0.0 | 594 | 0 |
| run14 | 980 | 0.0 | 0.0 | 0.0 | 980 | 0 |
| run15 | 602 | 0.0 | 0.0 | 0.0 | 602 | 0 |
| run22 | 1239 | 3.74 | 0.0 | 52.632 | 1065 | 174 |
| run23 | 644 | 7.527 | 0.0 | 29.268 | 452 | 192 |
| run24 | 278 | 14.948 | 7.368 | 23.469 | 11 | 267 |
| run25 | 177 | 24.995 | 18.644 | 32.558 | 0 | 177 |
| run50 | 763 | 22.456 | 5.263 | 43.902 | 16 | 747 |
| run51 | 865 | 31.279 | 10.256 | 62.791 | 0 | 865 |
| run58 | 659 | 10.072 | 0.0 | 30.769 | 337 | 322 |
| run59 | 291 | 17.454 | 10.377 | 23.585 | 0 | 291 |
| run60 | 198 | 27.466 | 22.481 | 32.716 | 0 | 198 |
| run61 | 734 | 18.723 | 2.564 | 52.5 | 45 | 689 |
| run62 | 386 | 34.982 | 20.833 | 46.739 | 0 | 386 |
| run63 | 284 | 47.439 | 37.5 | 57.48 | 0 | 284 |

## Figure Files


- `results/figures/training_curves.png`

- `results/figures/prediction_vs_actual.png`

- `results/figures/residual_histogram.png`

- `results/figures/confusion_matrix.png`

- `results/figures/feature_importance.png`

- `results/figures/results.png`
