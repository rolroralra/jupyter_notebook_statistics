## Jupyter Notebook URL
[https://jupyter.rolroralra.com](https://jupyter.rolroralra.com)

---
## How to Setup Jupyter Notebook Server as a daemon service
  - [https://towshif.github.io/site/tutorials/Python/setup-Jupyter/](https://towshif.github.io/site/tutorials/Python/setup-Jupyter/)
  - [https://program-error-review.tistory.com/14](https://program-error-review.tistory.com/14)
  - [https://goodtogreate.tistory.com/entry/IPython-Notebook-%EC%84%A4%EC%B9%98%EB%B0%A9%EB%B2%95](https://goodtogreate.tistory.com/entry/IPython-Notebook-%EC%84%A4%EC%B9%98%EB%B0%A9%EB%B2%95)

---
#### Issues
- nginx reverse proxy setting
	> [https://jupyterhub.readthedocs.io/en/stable/reference/config-proxy.html](https://jupyterhub.readthedocs.io/en/stable/reference/config-proxy.html)
- allow remote host setting in Jupyter
	> [https://github.com/jupyterhub/jupyterhub/issues/2230](https://github.com/jupyterhub/jupyterhub/issues/2230)
- jupyter, tornado version conflict
	> [https://github.com/jupyter/notebook/issues/4439](https://github.com/jupyter/notebook/issues/4439)

<details>
  <summary>Details</summary>
  <p>

---
#### /etc/nginx/conf.d/jupyter.conf
```nginx.conf
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

server {
	listen       443 ssl http2;
	listen       [::]:443 ssl http2;
	server_name  jupyter.rolroralra.com;
	#root         /usr/share/nginx/html;

	# Load configuration files for the default server block.
	#include /etc/nginx/default.d/*.conf;
	include /etc/nginx/default.d/certbot_ssl.conf;	# managed by Certbot

  add_header Strict-Transport-Security max-age=15768000;

	# Managing literal requests to the JupyterHub front end
	location / {
		proxy_pass       http://localhost:8888;
		proxy_set_header X-Real-IP $remote_addr;
		proxy_set_header Host $host;
		proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

		# websocket headers
		proxy_http_version 1.1;
		proxy_set_header Upgrade $http_upgrade;
		proxy_set_header Connection $connection_upgrade;

		proxy_buffering off;
	}

 	#Managing requests to verify letsencrypt host
 	location ~ /.well-known {
 		allow all;
 	}


	error_page 404 /404.html;
		location = /40x.html {
	}

	error_page 500 502 503 504 /50x.html;
		location = /50x.html {
	}
}
```
  </p>
</details>

---
### 통계학 필기노트
- [Probability Distribution](https://github.com/rolroralra/jupyter_notebook_statistics/blob/master/workspace/Probability_Distribution.ipynb)
- [Sampling Distribution](https://github.com/rolroralra/jupyter_notebook_statistics/blob/master/workspace/Sampling_Distribution.ipynb)
- [Inferential Statistics](https://github.com/rolroralra/jupyter_notebook_statistics/blob/master/workspace/Inferential_Statistics.ipynb)
- [Inferential Statistics 2](https://github.com/rolroralra/jupyter_notebook_statistics/blob/master/workspace/Inferential_Statistics_2.ipynb)

```ipython
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import scipy.stats as stats"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Description|symbol\n",
    ":----------|:-----\n",
    "Parameter (모수)|$\\theta$\n",
    "Statistic (통계량) - 표본으로부터 계산되는 Random Variable|$\\hat{\\theta}$\n",
    "Estimator (추정량) - 모수를 추정하는데 사용하는 통계량|$\\hat{\\theta}$\n",
    "Estimate (추정치, 추정값) - 추정량의 관측값|$\\hat{\\theta_{0}}$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Inferential Statistics (추론통계학)\n",
    "## Estimation (추정)\n",
    "* Point Estimation (점 추정)\n",
    "* Interval Estimation (구간 추정)\n",
    "\n",
    "### Confidence Interval (신뢰구간)\n",
    "> $P[\\hat{\\theta_L} \\le \\theta \\le \\hat{\\theta_U}]=1-\\alpha$\n",
    ">\n",
    "> $[\\hat{\\theta_L},\\hat{\\theta_U}]\\,:\\,100(1-\\alpha)\\%\\,Confidence Interval$\n",
    "\n",
    "* Example\n",
    "    > $\\begin{aligned}\\bar{X}-z_{\\alpha\\,/\\,2}\\,\\frac{\\sigma}{\\sqrt{n}}\\, \\le \\, \\mu \\, \\le \\, \\bar{X}+z_{\\alpha\\,/\\,2}\\,\\frac{\\sigma}{\\sqrt{n}}\\end{aligned}$    \n",
    "    >\n",
    "    > $\\begin{aligned}\\bar{X}-z_{\\alpha\\,/\\,2}\\,\\frac{\\sigma}{\\sqrt{n}} \\, \\le \\mu \\le \\bar{X}+z_{\\alpha\\,/\\,2}\\,\\frac{\\sigma}{\\sqrt{n}}\\end{aligned}$    \n",
    "    >\n",
    "    > $\\begin{aligned}\\frac{(n-1)}{\\chi^{2}_{\\frac{\\alpha}{2},n-1}}S^2 \\, \\le \\, \\sigma^2 \\, \\le \\, \\frac{(n-1)}{\\chi^{2}_{1-\\frac{\\alpha}{2},n-1}}S^2\\end{aligned}$    \n",
    "\n",
    "### Confidence Level (신뢰수준)\n",
    "> $1-\\alpha$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **Generalized Theorem** --- $\\bar{X}$\n",
    "\n",
    "> If $\\begin{aligned}X\\end{aligned}$ ~ $\\begin{aligned}N(\\mu, \\sigma^2)\\end{aligned}$, then\n",
    "> $\\begin{aligned}\\bar{X}\\end{aligned}$ ~ $\\begin{aligned}N(\\mu,\\frac{\\sigma^2}{n})\\end{aligned}$<br/>\n",
    ">\n",
    "> $\\begin{aligned}Z = \\frac{\\bar{X}-\\mu}{\\frac{\\sigma}{\\sqrt{n}}}\\end{aligned}$ ~ $\\begin{aligned}N(0,1)\\end{aligned}$\n",
    "\n",
    "#### Confidence Interval $(1-\\alpha)$\n",
    "> $\\begin{aligned} {\\bar{X}-z_{\\alpha\\,/\\,2}\\,\\frac{\\sigma}{\\sqrt{n}}},{\\bar{X}+z_{\\alpha\\,/\\,2}\\,\\frac{\\sigma}{\\sqrt{n}}} \\end{aligned}$\n",
    ">\n",
    "> $\\begin{aligned}\\bar{X}-z_{\\alpha\\,/\\,2}\\,\\frac{\\sigma}{\\sqrt{n}}\\, \\le \\, \\mu \\, \\le \\, \\bar{X}+z_{\\alpha\\,/\\,2}\\,\\frac{\\sigma}{\\sqrt{n}}\\end{aligned}$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(171.81778872525445, 174.56173830361053)\n",
      "(171.81778872525445, 174.56173830361053)\n"
     ]
    }
   ],
   "source": [
    "sample_size = 100\n",
    "n = sample_size\n",
    "population_mean = 173\n",
    "population_std = 7\n",
    "confidence_level = 0.95\n",
    "\n",
    "\n",
    "stats.norm.random_state=123\n",
    "\n",
    "sample = stats.norm.rvs(loc=population_mean, scale=population_std, size=n)\n",
    "\n",
    "sample_mean = np.mean(sample)\n",
    "sample_var = np.var(sample, ddof=1)\n",
    "sample_std = np.std(sample, ddof=1)\n",
    "\n",
    "\n",
    "alpha = 1 - confidence_level\n",
    "\n",
    "z_alpha = stats.norm.isf(alpha / 2)\n",
    "# z_alpha = stats.norm.ppf(1 - alpha / 2)\n",
    "\n",
    "delta = z_alpha * population_std / (n ** 0.5)\n",
    "\n",
    "confidence_interval = (sample_mean - delta, sample_mean + delta)\n",
    "\n",
    "print(confidence_interval)\n",
    "print(stats.norm.interval(confidence_level, loc = sample_mean, scale = population_std / (n ** 0.5)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### **Central Limit Theorem (CLT)** --- $\\bar{X}$\n",
    "\n",
    "> If $\\begin{aligned}E(X)=\\mu\\end{aligned}$, $\\begin{aligned}V(X)=\\sigma^2\\end{aligned}$, $\\begin{aligned}n \\gt 30\\end{aligned}$, then \n",
    "> $\\begin{aligned}\\bar{X}\\end{aligned}$ ~ $\\begin{aligned}N(\\mu,\\frac{\\sigma^2}{n})\\end{aligned}$<br/>\n",
    ">\n",
    "> $\\begin{aligned}Z = \\frac{\\bar{X}-\\mu}{\\frac{\\sigma}{\\sqrt{n}}}\\end{aligned}$ ~ $\\begin{aligned}N(0,1)\\end{aligned}$\n",
    "\n",
    "#### Confidence Interval $(1-\\alpha)$\n",
    "> $\\begin{aligned}\\bar{X}-z_{\\alpha\\,/\\,2}\\,\\frac{\\sigma}{\\sqrt{n}},\\,\\bar{X}+z_{\\alpha\\,/\\,2}\\,\\frac{\\sigma}{\\sqrt{n}}\\end{aligned}$\n",
    ">\n",
    "> $\\begin{aligned}\\bar{X}-z_{\\alpha\\,/\\,2}\\,\\frac{\\sigma}{\\sqrt{n}} \\, \\le \\mu \\le \\bar{X}+z_{\\alpha\\,/\\,2}\\,\\frac{\\sigma}{\\sqrt{n}}\\end{aligned}$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### **Sample Variance Distribution** --- $S^2$\n",
    "\n",
    "> If $X$ ~ $N(\\mu, \\sigma^2)$, then\n",
    ">\n",
    "> $\\begin{aligned}\\frac{(n\\,-\\,1)\\,S^2}{\\sigma^2}\\end{aligned}$ ~ $\\begin{aligned}\\chi^2(n-1)\\end{aligned}$<br/>\n",
    "\n",
    "#### Confidence Interval $(1-\\alpha)$\n",
    "\n",
    "[https://learn.lboro.ac.uk/archive/olmp/olmp_resources/pages/workbooks_1_50_jan2008/Workbook40/40_2_intvl_est_var.pdf](https://learn.lboro.ac.uk/archive/olmp/olmp_resources/pages/workbooks_1_50_jan2008/Workbook40/40_2_intvl_est_var.pdf)\n",
    "\n",
    "> $\\begin{aligned}\\frac{(n-1)}{\\chi^{2}_{\\frac{\\alpha}{2},n-1}}\\cdot S^{2},\\frac{(n-1)}{\\chi^{2}_{1-\\frac{\\alpha}{2},n-1}}\\cdot S^{2}\\end{aligned}$\n",
    ">\n",
    "> $\\begin{aligned}\\frac{(n-1)}{\\chi^{2}_{\\frac{\\alpha}{2},n-1}}\\cdot S^{2} \\, \\le \\, \\sigma^2 \\, \\le \\, \\frac{(n-1)}{\\chi^{2}_{1-\\frac{\\alpha}{2},n-1}}\\cdot S^{2}\\end{aligned}$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "128.4219886438403 73.36108019128368\n",
      "(48.56909790968567, 85.02246864326166)\n",
      "(48.56909790968566, 85.02246864326166)\n"
     ]
    }
   ],
   "source": [
    "sample_size = 100\n",
    "n = sample_size\n",
    "population_mean = 173\n",
    "population_std = 7\n",
    "confidence_level = 0.95\n",
    "\n",
    "stats.norm.random_state=123\n",
    "\n",
    "sample = stats.norm.rvs(loc=population_mean, scale=population_std, size=n)\n",
    "\n",
    "sample_mean = np.mean(sample)\n",
    "sample_var = np.var(sample, ddof=1)\n",
    "sample_std = np.std(sample, ddof=1)\n",
    "\n",
    "\n",
    "alpha = 1 - confidence_level\n",
    "\n",
    "# chi2_alpah = stats.chi2.isf(alpha/2, n - 1)\n",
    "# chi2_1_minus_alpah = stats.chi2.ppf(alpha/2, n - 1)\n",
    "chi2_1_minus_alpah, chi2_alpah = stats.chi2.interval(confidence_level, df=n - 1)\n",
    "\n",
    "print(chi2_alpah, chi2_1_minus_alpah)\n",
    "\n",
    "confidence_interval = (sample_var * (n - 1) / chi2_alpah, sample_var * (n - 1) / chi2_1_minus_alpah)\n",
    "print(confidence_interval)\n",
    "\n",
    "confidence_interval_2 = (np.array(stats.chi2.interval(confidence_level, n - 1))[::-1] ** -1) * (sample_var * (n - 1))\n",
    "print(tuple(confidence_interval_2))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### **Sample Mean and Sample Variance** --- $\\bar{X},\\,S^2$\n",
    "\n",
    "> If $X$ ~ $N(\\mu, \\sigma^2)$, then\n",
    ">\n",
    "> $\\begin{aligned}\\frac{\\bar{X}-\\mu}{\\frac{S}{\\sqrt{n}}}\\end{aligned}$ ~ $\\begin{aligned}t(n-1)\\end{aligned}$<br/>\n",
    "\n",
    "#### Confidence Interval $(1-\\alpha)$\n",
    "> $\\begin{aligned}\\bar{X}-t_{\\alpha\\,/\\,2,\\,n-1}\\,\\frac{S}{\\sqrt{n}},\\,\\bar{X}+t_{\\alpha\\,/\\,2,\\,n-1}\\,\\frac{S}{\\sqrt{n}}\\end{aligned}$\n",
    ">\n",
    "> $\\begin{aligned}\\bar{X}-t_{\\alpha\\,/\\,2,\\,n-1}\\,\\frac{S}{\\sqrt{n}} \\le \\mu \\le \\,\\bar{X}+t_{\\alpha\\,/\\,2,\\,n-1}\\,\\frac{S}{\\sqrt{n}}\\end{aligned}$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 245,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(171.61479718984518, 174.7647298390198)\n",
      "(171.61479718984518, 174.7647298390198)\n"
     ]
    }
   ],
   "source": [
    "sample_size = 100\n",
    "n = sample_size\n",
    "population_mean = 173\n",
    "population_std = 7\n",
    "confidence_level = 0.95\n",
    "\n",
    "\n",
    "stats.norm.random_state=123\n",
    "\n",
    "sample = stats.norm.rvs(loc=population_mean, scale=population_std, size=n)\n",
    "\n",
    "sample_mean = np.mean(sample)\n",
    "sample_var = np.var(sample, ddof=1)\n",
    "sample_std = np.std(sample, ddof=1)\n",
    "\n",
    "\n",
    "alpha = 1 - confidence_level\n",
    "\n",
    "t_alpha = stats.t.isf(alpha/2,n-1)\n",
    "# t_alpha = stats.t.ppf(1 - alpha / 2,n-1)\n",
    "\n",
    "delta = t_alpha * sample_std / (n ** 0.5)\n",
    "\n",
    "confidence_interval = (sample_mean - delta, sample_mean + delta)\n",
    "\n",
    "print(confidence_interval)\n",
    "print(stats.t.interval(confidence_level, n - 1, loc=sample_mean, scale=sample_std / (n ** 0.5)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### **Sample Ratio** --- $\\begin{aligned}\\hat{p} \\equiv \\frac{\\bar{X}}{N}\\end{aligned}$\n",
    "\n",
    "> If $X$ ~ $Bin(N,p)$, $N\\gt30$ and sample_size is \"n\" then\n",
    ">\n",
    "> $\\begin{aligned}\\hat{p}\\end{aligned}$ ~ $\\begin{aligned}N(p, \\frac{pq}{nN})\\end{aligned}$<br/>\n",
    ">\n",
    "> $\\begin{aligned}\\frac{\\hat{p}-p}{\\sqrt{\\frac{pq}{nN}}}\\end{aligned}$ ~ $\\begin{aligned}N(0,1)\\end{aligned}$\n",
    "\n",
    "#### Confidence Interval $(1-\\alpha)$\n",
    "> $\\begin{aligned}\\,\\hat{p}-z_{\\alpha\\,/\\,2}\\,\\sqrt{\\frac{pq}{nN}},\\,\\hat{p}+z_{\\alpha\\,/\\,2}\\,\\sqrt{\\frac{pq}{nN}}\\,\\end{aligned}$\n",
    ">\n",
    "> $\\begin{aligned}\\hat{p}-z_{\\alpha\\,/\\,2}\\,\\sqrt{\\frac{pq}{nN}} \\le p \\le \\,\\hat{p}+z_{\\alpha\\,/\\,2}\\,\\sqrt{\\frac{pq}{nN}}\\end{aligned}$"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 352,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X ~ Bin(100, 0.3)\n",
      "population_ratio: 0.3\n",
      "\n",
      "sample_size: 100\n",
      "estimate_ratio: 0.3007\n",
      "confidence_interval: (0.291718316681458, 0.30968168331854207)\n",
      "confidence_interval: (0.291718316681458, 0.30968168331854207)\n"
     ]
    }
   ],
   "source": [
    "# n = np.random.randint(30, 200)   # sample size\n",
    "n = 100\n",
    "N = 100                          # binomial parameter, N\n",
    "population_ratio = 0.3\n",
    "p = population_ratio\n",
    "confidence_level = 0.95\n",
    "\n",
    "print(f\"X ~ Bin({N}, {p})\")\n",
    "print(f\"population_ratio: {p}\")\n",
    "print()\n",
    "\n",
    "alpha = 1 - confidence_level\n",
    "\n",
    "stats.binom.random_state = 123\n",
    "\n",
    "X = stats.binom.rvs(N, p, size=n)\n",
    "\n",
    "estimate_ratio = np.mean(X) / N\n",
    "\n",
    "print(f\"sample_size: {n}\")\n",
    "print(f\"estimate_ratio: {estimate_ratio}\")\n",
    "\n",
    "delta = stats.norm.isf(alpha / 2) * ((p * (1 - p) / N / n) ** 0.5)\n",
    "confidence_interval = (estimate_ratio - delta, estimate_ratio + delta)\n",
    "# confidence_interval = stats.norm.interval(confidence_level, loc=estimate_ratio, scale=((((p * (1 - p) / N / n) ** 0.5))))\n",
    "\n",
    "print(f\"confidence_interval: {confidence_interval}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Hypothesis Test (가설 검정)\n",
    "> 표본으로부터 주어지는 정보를 이용하여, 모수에 대한 예상, 주장 또는 추측 등의 옳고 그름을 판정하는 과정\n",
    "\n",
    "* **Null Hypothesis (귀무 가설):** $H_0$\n",
    "> 지금까지 사실로 알려져 있는 가설, 대립가설이 참이라는 확실한 근거가 없을 때 받아들이는 가설\n",
    "\n",
    "* **Alternative Hypothesis (대립 가설):** $H_1$\n",
    "> 표본자료로부터의 강력한 증거에 의해 입증하고자 하는 가설\n",
    "\n",
    "### Test Statistic (검정통계량)\n",
    "> 가설 검정에 사용되는 통계량(Random Variable)\n",
    "\n",
    "### 귀무 가설 $H_0$  기각여부 판단하는 2가지 방법\n",
    "> 1. **Rejection Region(기각역)에 의한 검정**\n",
    "> 2. **p-value(유의확률)에 의한 검정**\n",
    "\n",
    "### Rejection Region (기각역)\n",
    "> 귀무가설 $H_0$을 기각하게 하는 검정통계량 관측치의 영역\n",
    "\n",
    "### Critical Value (임계치)\n",
    "> 기각역의 경계가 되는 값\n",
    "\n",
    "### Significance Level (유의수준)\n",
    "> 제1종 오류를 범할 확률의 최대 허용 한계, $\\alpha$\n",
    "\n",
    "### p-value (유의확률)\n",
    "> 검정통계량 $\\hat{\\theta}$의 관찰값 $\\hat{\\theta_0}$에 대하여 귀무가설 $H_0$를 기각할 수 있는 최소한의 유의수준으로 정의됨.\n",
    ">\n",
    ">  $p$-$value\\le \\alpha$ 면, $H_0$를 기각"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sample_mean: 173.18976351443249\n",
      "\n",
      "test-statistic: 0.271 for Mean by using Standard Normal Distribution\n",
      "p-value: 0.786\n",
      "\n",
      "0.786 > 0.05, H_0 is not rejected\n",
      "\n",
      "0.3931606236689139\n",
      "0.6068393763310861\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "NormaltestResult(statistic=2.7252176285631196, pvalue=0.25599206929942614)"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "sample_size = 100\n",
    "n = sample_size\n",
    "population_mean = 173\n",
    "population_std = 7\n",
    "significance_level = 0.05\n",
    "\n",
    "\n",
    "stats.norm.random_state=123\n",
    "\n",
    "sample = stats.norm.rvs(loc=population_mean, scale=population_std, size=n)\n",
    "\n",
    "sample_mean = np.mean(sample)\n",
    "sample_var = np.var(sample, ddof=1)\n",
    "sample_std = np.std(sample, ddof=1)\n",
    "\n",
    "print(f\"sample_mean: {sample_mean}\")\n",
    "print()\n",
    "\n",
    "test_statistic_value = (sample_mean - population_mean) / population_std * np.sqrt(n)\n",
    "p_value = np.min([stats.norm.cdf(test_statistic_value),stats.norm.sf(test_statistic_value)]) * 2\n",
    "p_value_larger = stats.norm.sf(test_statistic_value)\n",
    "p_value_smaller = stats.norm.cdf(test_statistic_value)\n",
    "\n",
    "# print(sample_mean)\n",
    "# print(sample_std)\n",
    "print(f\"test-statistic: {round(test_statistic_value, 3)} for Mean by using Standard Normal Distribution\")\n",
    "print(f\"p-value: {round(p_value,3)}\")\n",
    "print()\n",
    "\n",
    "if p_value > significance_level:\n",
    "    print(f\"{round(p_value,3)} > {round(significance_level, 3)}, H_0 is not rejected\")\n",
    "else:\n",
    "    print(f\"{round(p_value,3)} <= {round(significance_level, 3)}, H_0 is rejected\")\n",
    "\n",
    "print()\n",
    "print(p_value_larger)\n",
    "print(p_value_smaller)\n",
    "\n",
    "\n",
    "# stats.normaltest((sample - sample_mean)/sample_std)\n",
    "stats.normaltest(sample)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sample_mean: 173.18976351443249\n",
      "\n",
      "test-statistic: 0.239 for Mean by using t Distribution\n",
      "p-value: 0.812\n",
      "\n",
      "test-statistic: 0.239 for Mean by using t Distribution\n",
      "p-value: 0.812\n",
      "\n",
      "0.812 > 0.05, H_0 is not rejected\n",
      "\n",
      "0.40577158682490666\n",
      "0.5942284131750933\n"
     ]
    }
   ],
   "source": [
    "sample_size = 100\n",
    "n = sample_size\n",
    "population_mean = 173\n",
    "population_std = 7\n",
    "significance_level = 0.05\n",
    "\n",
    "\n",
    "stats.norm.random_state=123\n",
    "\n",
    "sample = stats.norm.rvs(loc=population_mean, scale=population_std, size=n)\n",
    "\n",
    "sample_mean = np.mean(sample)\n",
    "sample_var = np.var(sample, ddof=1)\n",
    "sample_std = np.std(sample, ddof=1)\n",
    "\n",
    "dof = n - 1\n",
    "\n",
    "print(f\"sample_mean: {sample_mean}\")\n",
    "print()\n",
    "\n",
    "test_statistic_value = (sample_mean - population_mean) / sample_std * np.sqrt(n)\n",
    "p_value = np.min([stats.t.cdf(test_statistic_value, dof),stats.t.sf(test_statistic_value, dof)]) * 2\n",
    "# test_statistic_value, p_value = stats.ttest_1samp(sample, popmean=population_mean)\n",
    "p_value_larger = stats.t.sf(test_statistic_value, dof)\n",
    "p_value_smaller = stats.t.cdf(test_statistic_value, dof)\n",
    "\n",
    "\n",
    "# print(sample_mean)\n",
    "# print(sample_std)\n",
    "print(f\"test-statistic: {round(test_statistic_value, 3)} for Mean by using t Distribution\")\n",
    "print(f\"p-value: {round(p_value,3)}\")\n",
    "print()\n",
    "print(f\"test-statistic: {round(stats.ttest_1samp(sample, popmean=population_mean)[0], 3)} for Mean by using t Distribution\")\n",
    "print(f\"p-value: {round(stats.ttest_1samp(sample, popmean=population_mean)[1],3)}\")\n",
    "print()\n",
    "\n",
    "if p_value > significance_level:\n",
    "    print(f\"{round(p_value,3)} > {round(significance_level, 3)}, H_0 is not rejected\")\n",
    "else:\n",
    "    print(f\"{round(p_value,3)} <= {round(significance_level, 3)}, H_0 is rejected\")\n",
    "    \n",
    "print()\n",
    "print(p_value_larger)\n",
    "print(p_value_smaller)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "sample_variance: 63.003435759588086\n",
      "\n",
      "test-statistic: 127.293 for Variance by using Chi-squared Distribution\n",
      "p-value: 0.058\n",
      "\n",
      "0.058 > 0.05, H_0 is not rejected\n",
      "\n",
      "0.029231874004325167\n",
      "0.9707681259956749\n"
     ]
    }
   ],
   "source": [
    "sample_size = 100\n",
    "n = sample_size\n",
    "population_mean = 173\n",
    "population_std = 7\n",
    "significance_level = 0.05\n",
    "\n",
    "\n",
    "stats.norm.random_state=123\n",
    "\n",
    "sample = stats.norm.rvs(loc=population_mean, scale=population_std, size=n)\n",
    "\n",
    "sample_mean = np.mean(sample)\n",
    "sample_var = np.var(sample, ddof=1)\n",
    "sample_std = np.std(sample, ddof=1)\n",
    "\n",
    "dof = n - 1\n",
    "\n",
    "print(f\"sample_variance: {sample_var}\")\n",
    "print()\n",
    "\n",
    "test_statistic_value = (n - 1) * sample_var / (population_std ** 2)\n",
    "p_value = np.min([stats.chi2.cdf(test_statistic_value, dof),stats.chi2.sf(test_statistic_value, dof)]) * 2\n",
    "p_value_larger = stats.chi2.sf(test_statistic_value, dof)\n",
    "p_value_smaller = stats.chi2.cdf(test_statistic_value, dof)\n",
    "\n",
    "# print(sample_mean)\n",
    "# print(sample_std)\n",
    "print(f\"test-statistic: {round(test_statistic_value, 3)} for Variance by using Chi-squared Distribution\")\n",
    "print(f\"p-value: {round(p_value,3)}\")\n",
    "print()\n",
    "\n",
    "if p_value > significance_level:\n",
    "    print(f\"{round(p_value,3)} > {round(significance_level, 3)}, H_0 is not rejected\")\n",
    "else:\n",
    "    print(f\"{round(p_value,3)} <= {round(significance_level, 3)}, H_0 is rejected\")\n",
    "\n",
    "print()\n",
    "print(p_value_larger)\n",
    "print(p_value_smaller)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "venvPy3",
   "language": "python",
   "name": "venvpy3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
```
