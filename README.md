# Moving-Horizon-Estimator
Study on moving horizon estimator

아래에서는 **수식 단계별로** 보여드릴게요 — (1) 모델과 가정, (2) 칼만필터가 구하는 것(MMSE = posterior mean), (3) 배치(MAP) 문제(=MHE의 기반)의 음의 로그(2차식) 전개, (4) 가우시안에서 MAP = MMSE가 되는 이유, (5) arrival-cost(창 시작 prior)를 정확히 주었을 때 MHE(MAP)가 KF의 posterior mean과 같아지는 논리적 동치 증명. 핵심은 **가우시안 가정 → posterior가 가우시안 → MAP(최대사후확률) = posterior 평균(MMSE)** 이고, MHE의 objective가 정확한 prior을 포함하면 같은 음의 로그우도(=같은 2차식)를 최적화한다는 점입니다.

---

## 0. 표기 · 모델 · 가정

이산시간 선형 모델:
[
\begin{aligned}
x_{k+1} &= A x_k + B u_k + w_k,\qquad w_k\sim\mathcal N(0,Q),\
y_k &= C x_k + D u_k + v_k,\qquad v_k\sim\mathcal N(0,R).
\end{aligned}
]
초기 prior: (x_0\sim\mathcal N(\bar x_0,P_0)).
(w_k,v_k)는 서로 독립이고 시간간 독립(white Gaussian)이라 가정합니다. 이 가정이 핵심입니다.

---

## 1. 칼만필터(KF)가 목표로 하는 것 — MMSE (조건부 평균)

칼만필터는 시점 (k)에서의 **조건부 평균 (MMSE)** 을 구합니다:
[
\hat x_{k|k}^{\text{KF}} ;=; \mathbb{E}[,x_k \mid Y_{0:k},],
\quad\text{여기서 }Y_{0:k}={y_0,\dots,y_k}.
]
확률론적 정의로는
[
\hat x_{k|k}^{\text{KF}} ;=; \arg\min_{x}; \mathbb{E}\big[|x_k - x|^2 \mid Y_{0:k}\big],
]
즉, posterior (p(x_k\mid Y_{0:k}))의 평균입니다.

---

## 2. 배치 MAP (full-information MAP) — 음의 로그 우도(negative log posterior)

배치 MAP은 전체 궤적 (X_{0:k}=[x_0,\dots,x_k])에 대해 posterior를 최대화합니다.
베이즈 법칙과 독립성으로 결합우도는
[
p(X_{0:k},Y_{0:k}) = p(x_0)\prod_{i=0}^{k-1} p(x_{i+1}\mid x_i,u_i)\prod_{i=0}^{k} p(y_i\mid x_i,u_i).
]
가우시안 가정 하에서 각 확률밀도는 지수형(정규밀도)이고, 음의 로그를 취하면(상수항은 무시)
[
\begin{aligned}
J_{\text{full}}(X_{0:k})
&= -\log p(X_{0:k}\mid Y_{0:k}) + \text{const} \
&= (x_0-\bar x_0)^\top P_0^{-1}(x_0-\bar x_0) \
&\quad + \sum_{i=0}^{k-1} (x_{i+1}-A x_i - B u_i)^\top Q^{-1} (x_{i+1}-A x_i - B u_i)\
&\quad + \sum_{i=0}^{k} (y_i - C x_i - D u_i)^\top R^{-1} (y_i - C x_i - D u_i).
\end{aligned}
]
이는 **이차(quadratic)** 함수입니다. 배치 MAP은
[
\hat X_{0:k}^{\text{MAP}} = \arg\min_{X_{0:k}} J_{\text{full}}(X_{0:k}).
]

이제 중요한 점: **가우시안 모델에서 posterior (p(X_{0:k}\mid Y_{0:k}))는 가우시안**이며, 그 평균(벡터)은 위 2차식을 최소화하는 해입니다(표준 사실: 가우시안의 모드 = 평균). 따라서 배치 문제의 해는 posterior mean을 줍니다.

특히 (x_k) 관점에서는,
[
\hat x_k^{\text{batch-MAP}} ;=; \text{the }x_k\text{ component of }\hat X_{0:k}^{\text{MAP}}.
]
그리고 가우시안이므로 이것은 ( \mathbb{E}[x_k\mid Y_{0:k}] )와 같습니다. 즉 **batch-MAP의 (x_k) = KF의 posterior mean**(둘 다 전체 관측 (Y_{0:k})을 동등하게 고려함).

---

## 3. MHE(Moving-Horizon MAP)와 arrival cost

MHE는 **최근 (N) 스텝만** 고려하는 MAP입니다. 시점 (k)에서 변수는 (X_{k-N:k}=[x_{k-N},\dots,x_k]). 창 이전(즉 (0!:!k!-!N))의 정보는 하나의 prior로 요약됩니다 — 이것이 **arrival cost**입니다.

MHE의 음의 로그우도(또는 MAP 목적함수)는
[
\begin{aligned}
J_{\text{MHE}}(X_{k-N:k})
&= -\log p(x_{k-N}\mid Y_{0:k-N}) \
&\quad + \sum_{i=k-N}^{k-1} (x_{i+1}-A x_i - B u_i)^\top Q^{-1} (x_{i+1}-A x_i - B u_i) \
&\quad + \sum_{i=k-N}^{k} (y_i - C x_i - D u_i)^\top R^{-1} (y_i - C x_i - D u_i),
\end{aligned}
]
여기서 첫항이 **arrival cost**로, 보통 가우시안 가정 하에서
[
-\log p(x_{k-N}\mid Y_{0:k-N}) ;\propto; (x_{k-N}-\bar x_{k-N})^\top P_a^{-1}(x_{k-N}-\bar x_{k-N})
]
즉 (\bar x_{k-N},P_a)가 arrival mean & covariance입니다.

---

## 4. conditional independence로부터의 분해(중요)

Markov 성질과 베이즈 법칙에 의해(정확한 전개)
[
p(X_{k-N:k}\mid Y_{0:k}) ;\propto; p(x_{k-N}\mid Y_{0:k-N}); \prod_{i=k-N}^{k-1} p(x_{i+1}\mid x_i,u_i); \prod_{i=k-N}^{k} p(y_i\mid x_i,u_i).
]
따라서 MHE의 목적함수 (J_{\text{MHE}}(X_{k-N:k}))는 **정확한 arrival prior** (p(x_{k-N}\mid Y_{0:k-N}))를 사용하면 **음의 로그 posterior of (X_{k-N:k}) given (Y_{0:k})** 과 정확히 동등합니다(상수 차이만). 즉,
[
J_{\text{MHE}}(X_{k-N:k}) = -\log p(X_{k-N:k}\mid Y_{0:k}) + \text{const},
]
when the arrival cost is exactly (-\log p(x_{k-N}\mid Y_{0:k-N})).

따라서 **최적화의 해 (\arg\min J_{\text{MHE}})** 는 (p(X_{k-N:k}\mid Y_{0:k}))의 MAP(모드)을 구하는 것과 동일합니다.

---

## 5. 가우시안이면 MAP = MMSE

가우시안 분포 (\mathcal N(\mu,\Sigma))에 대해 밀도는
[
p(x)\propto \exp\big(-\tfrac12 (x-\mu)^\top \Sigma^{-1}(x-\mu)\big).
]
음의 로그(제거상수)는 (\tfrac12 (x-\mu)^\top \Sigma^{-1}(x-\mu)). 이 2차식의 최소값을 주는 점은 (x=\mu). 즉 **모드(MAP) = 평균(=MMSE)** 입니다. 따라서 posterior가 가우시안이면 MAP과 posterior mean은 같고, 이 값이 바로 KF가 계산하는 (\mathbb{E}[x_k\mid Y_{0:k}])입니다.

---

## 6. 결론(동치 조건과 요약)

* **전체(배치) MAP**을 풀면 posterior (p(X_{0:k}\mid Y_{0:k}))의 평균(모든 시점의 posterior mean 벡터)을 얻게 되고, 그 중 (x_k) 성분은 KF가 주는 (\hat x_{k|k})와 동일합니다(선형·가우시안 조건 하에서).
* **MHE**는 최근 (N) 스텝만 최적화하지만, 만약 arrival cost를 **정확한 prior**로 즉
  [
  (x_{k-N}-\bar x_{k-N})^\top P_a^{-1}(x_{k-N}-\bar x_{k-N}) \quad\text{with}\quad
  \bar x_{k-N}=\hat x_{k-N|k-N},; P_a=P_{k-N|k-N},
  ]
  로 설정하면, MHE의 목적함수는 (p(X_{k-N:k}\mid Y_{0:k}))의 음의 로그와 같아지고, 그 최적해(특히 (x_k) 성분)는 **배치 MAP(따라서 KF posterior mean)** 과 동일합니다.
* 더 간단히 말하면:
  **선형·가우시안** + **arrival cost = 정확한 posterior at window start** 이면
  [
  \boxed{;\text{MHE(MAP)};=;\text{KF(MMSE)};(\text{동일한 }x_k);}.
  ]

---

## 부가적 표기 (정보형태로 한줄 요약)

음의 로그 posterior(배치)의 2차형식을 정보행렬·정보벡터로 쓰면
[
J_{\text{full}}(X) = \tfrac12 X^\top H X - b^\top X + \text{const},
]
최소해는 (H X^\star = b). Posterior covariance (=H^{-1}), posterior mean (=X^\star). MHE에 정확한 arrival prior을 주면 MHE의 (H,b)는 배치의 (H,b)의 부분블록과 일치하여 동일한 (x_k)를 줍니다.

---

원하시면, 위 논리를 **구체적 1-D/2-D 행렬 수치 예**(행렬 (A,B,C,Q,R)를 주고 직접 계산)로 보여드리겠습니다 — KF로 구한 (\hat x_{k-N|k-N},P_{k-N|k-N})를 arrival로 넣은 MHE가 어떻게 수치적으로 KF 결과와 완전히 일치하는지를 행렬 방정식(정보행렬 H와 정보벡터 b) 수준에서 보여드릴게요. 어느 차원을 원하시나요(1D/2D)?
