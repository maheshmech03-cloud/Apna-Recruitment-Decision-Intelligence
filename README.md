Domain: Recruitment

## How to run

Use Python 3.10 or newer. From the repository root after cloning the repository:

```bash
cd <repository-folder>
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

On Windows Command Prompt, activate the environment with `.venv\\Scripts\\activate` instead of the `source` command. `generate_data.py` recreates `data/records.csv`; it should print `800 10 170 0537bc7fb4a044719831c43166e79d3c`. The four Part scripts read that file and regenerate their committed results under `outputs/`.

## Part 1 findings

The sample mean is 8.9382 days, which is 0.9382 days above the planning assumption of 8.0. The test statistic is 2.8930, exceeding the two-tailed critical value 1.96, so I reject H0 at the 5% level. A Type I error here would mean concluding that the average offer-decision time differs from 8 days when the true population average is actually 8 days. The difference is close to one day; whether it is operationally material depends on hiring-volume and service-level tolerances, but it is large enough to review in planning rather than treating statistical significance alone as proof of business impact.

The required sample size for a 0.5-day margin at 95% confidence is 1,293, compared with 800 available records. The current dataset is therefore 493 records short of that target under the stated formula; the planning team should collect more observations before relying on that precision target.

Gradient descent's slope gap is 0.0000. Standardising the predictor gives a well-scaled convex least-squares objective, and the full-batch updates with the specified learning rate converge to the same unique least-squares line as the closed-form calculation.

The raw residual standard-deviation ratio is 3.7525, indicating a clear funnel: residual spread is much larger in the upper fitted-value half. The log-log model's ratio is 1.0396, so the log transformation largely removes the visible change in residual spread across the fitted-value groups.

Using the assignment's mechanical absolute-coefficient cutoff of 0.05, the qualifying Task 1.7 predictors are ln(X4) (1.1576), X2 (0.0940), X3 (0.1067), and X6 (0.1518). The adjusted R² rises from 0.6561 for the log-log model to 0.7403 for the eight-column multiple regression, a gain of 0.0842, so the additional predictors improve in-sample fit. Coefficients have different units and this cutoff is not a ranking of predictor importance.

## Part 2 findings

The 200-update losses from lowest to highest are η = 1.0 (0.3703), η = 0.1 (0.3776), and η = 0.01 (0.5091). In this run the smallest learning rate is too slow to reach the lower loss within 200 updates; the larger rate learns faster here, although an excessively large rate can overshoot or become unstable.

The three largest absolute logistic weights are X3 (0.9013), X1 (-0.5554), and X2 (0.5505). X3 has the largest odds ratio, 2.4629: holding the other standardised predictors fixed, a one-training-standard-deviation increase in the salary-gap ratio multiplies the modelled odds of an offer decline by about 2.46.

The always-negative baseline accuracy is 0.7875, close to the logistic model's 0.8042 accuracy at τ = 0.5. Accuracy alone is weak here because only 51 of 240 test records are positive, so a model can score well by mostly predicting the majority class. Lowering τ from 0.5 to 0.3 raises recall from 0.4706 to 0.7451 while reducing precision from 0.5455 to 0.4634.

At τ = 0.3, logistic regression has F1 = 0.5714 and recall = 0.7451; KNN (best K = 5) has F1 = 0.3696 and recall = 0.3333. Logistic regression is the stronger of these two test-set baselines on both measures.

The cost-minimising training threshold is τ = 0.15. On the test rows, logistic regression at that threshold costs INR 6,050, compared with INR 11,800 at τ = 0.5 and INR 20,400 if nobody is flagged. I would pilot logistic regression at τ = 0.15 with recruiter review, because it is the threshold selected using training costs and it substantially reduces the observed test cost. The threshold is selected on training rows rather than test rows to avoid using the held-out test outcomes to tune the decision rule and produce an optimistically biased evaluation.

## Part 3 findings

The successive WCSS drops are 659.8282 (K2 to K3), 108.0739 (K3 to K4), 171.0575 (K4 to K5), and 45.5466 (K5 to K6). The elbow rule selects K = 3 because the K3-to-K4 drop is less than half the preceding K2-to-K3 drop. The later drops do not shrink steadily; the pinned starting rows provide a reproducible solution, not necessarily the global optimum for each K.

- **Segment A — experienced, low-friction pipeline:** 322 records; mean experience 47.5028 months, 1.9503 reschedules, salary gap 0.2250, profile score 62.1584, and outcome rate 0.0186. Keep the process moving with timely status updates and avoid adding unnecessary interview steps.
- **Segment B — reschedule-heavy, mid-gap pipeline:** 255 records; mean experience 28.6071 months, 6.0706 reschedules, salary gap 0.4534, and outcome rate 0.2353. Coordinate interview availability earlier and confirm schedules to reduce repeated rescheduling.
- **Segment C — high-gap, early-experience pipeline:** 223 records; mean experience 14.3892 months, salary gap 0.7118, profile score 38.5215, and outcome rate 0.4664. Discuss compensation alignment early and help candidates complete missing profile information.

Segment C has the highest outcome rate, 0.4664. The segment names describe pipeline patterns, not personal characteristics.

At the dendrogram cut, single linkage gives {E1}, {E2, E3, E6}, {E4}, and {E5}; complete linkage gives {E1, E4, E5} and {E2, E3, E6}. Single linkage joins clusters when any one cross-cluster pair is close, so E1 remains separate from E4/E5 at the selected cut even though E4 and E5 are close. Complete linkage uses the most distant cross-cluster pair, delaying the final merge between the two broad groups and producing a different largest height gap and cut.

The standardised PCA matrix has a zero mean vector (up to floating-point rounding). PC1 is driven positively by X1 (0.5317) and X5 (0.4418), and negatively by X3 (-0.5187), X2 (-0.4100), and X4 (-0.2916); it contrasts experience/profile completeness with salary gap, rescheduling and expected salary. Three components are needed to retain at least 80% of the variance (cumulative ratio 0.8616). Mean PC1 scores are 1.7773 for A, -0.7831 for B and -1.6709 for C, showing a clear ordering of the three segments along this component.

## Part 4 findings

S4 (X1 ≤ 24) is best under both information gain and Gini gain in the supplied candidate splits. It is best because it produces the largest measured reduction in parent impurity among the four tested questions. Growing a tree until every leaf is pure would fit small-sample quirks and noise; a maximum-depth limit or a minimum number of samples per leaf can stop it from growing too deep.

I would ship logistic regression as the structured-data candidate for a controlled pilot: its Task 4.2 F1 is 0.5714 and recall is 0.7451. Combining the models did not help on this test set: hard voting reached F1 0.5344 and soft voting 0.5507, both below logistic regression. KNN and the stump also had lower F1 values (0.3696 and 0.5532 respectively).

After round 1 of AdaBoost, the two points misclassified by h1 (points 3 and 7) carry total normalised weight 0.5000. Round 2 trusts h2 more because its weighted error is lower (0.1667 versus 0.2500), giving it a larger coefficient (0.8047 versus 0.5493).

**Model A, option 2 — CNN layer shapes and parameter counts**

| Layer | Output shape | Parameters |
|---|---:|---:|
| Convolution 1 | 28 × 28 × 6 | 156 |
| Max-pool 1 | 14 × 14 × 6 | 0 |
| Convolution 2 | 10 × 10 × 16 | 2,416 |
| Max-pool 2 | 5 × 5 × 16 | 0 |
| Flatten | 400 | 0 |
| Dense 1 (ReLU) | 32 | 12,832 |
| Dense output | 10 | 330 |

The fully connected option (784 → 16 → 16 → 10) has 13,002 parameters; the CNN has 15,734. For the final 10-class layer I would use softmax with categorical cross-entropy. In the larger 128 × 128 × 3 comparison, the fully connected layer has 50,332,672 parameters while the 3 × 3 convolution with 64 filters has 1,792, illustrating the parameter efficiency of local weight sharing.

**Model B — recurrent resume-text classifier:** W_xh has shape 128 × 64, W_hh has shape 128 × 128, and W_hy has shape 16 × 128; the bias vectors have shapes 128 and 16. The total is 26,768 parameters. The hidden activation is tanh and the 16-class output would use softmax with categorical cross-entropy. The parameter count does not grow with sequence length because the same weight matrices and biases are reused at every time step; only the number of computations grows. A very deep plain network can suffer vanishing gradients, leaving early layers with weak learning signals; skip connections provide shorter paths for information and gradients to flow through the network.

## Model-choice memo

To: Head of Talent-Acquisition Operations

I recommend a controlled pilot of the logistic-regression model for structured candidate records, using the cost-selected threshold τ = 0.15. The training procedure selected this threshold without consulting the held-out test outcomes. On the 240 test records, the selected threshold produces an estimated cost of INR 6,050, compared with INR 11,800 at τ = 0.5 and INR 20,400 if no candidates are flagged. At τ = 0.15, test recall is 0.8824 (45 of 51 declines detected) and precision is 0.3814; recruiter review is therefore essential because some follow-ups will be unnecessary. At the standard τ = 0.3 comparison point, logistic regression achieved F1 = 0.5714, ahead of KNN at 0.3696. The principal evidence risk is that these are synthetic records and one fixed split, not a representative production evaluation. Do not build Model A or Model B before validating the structured workflow; resume-text Model B is the more relevant later extension once the structured model has been safely piloted.

## Limitations

1. The records are entirely synthetic and must not be interpreted as evidence about Apna's real candidates, products, policies or performance.
2. Evaluation uses one fixed stratified 70/30 split. A different sample or time period could produce different coefficients, thresholds and test metrics; repeated temporal and external validation would be needed.
3. The test set contains only 51 positive outcomes, so recall, precision and cost estimates have meaningful sampling uncertainty.
4. Logistic regression and the simple regression baselines impose functional-form assumptions; the KNN and clustering results also depend on the specified feature scaling, distance and initialisation rules.
5. No protected attributes are present, and no fairness assessment across protected groups is possible. These models are educational baselines and are not approved for automated candidate ranking or employment decisions.
