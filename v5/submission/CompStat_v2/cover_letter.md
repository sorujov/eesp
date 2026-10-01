Dear Editor,

I submit the manuscript "Group recovery after trimming, and level-free flagging, in robust
clusterwise regression" for consideration in *Computational Statistics*.

Trimming methods for clusterwise linear regression, of which TCLUST-REG is the best known, discard
a fixed fraction of the data. A level below the fraction of outliers breaks the fit, and a
generous level can trim a small group away together with the outliers. The manuscript makes two
proposals.

- A group-recovery step that can follow any trimming or flagging method. It searches the
  discarded units for a line and restores it as a group when the likelihood of Gaussian groups
  plus uniform noise prefers the fit that contains it. After TCLUST-REG and after the trimmed
  cluster-weighted model at a generous level with reweighting, it returned the small groups that
  the level had trimmed away.
- ESF, a flagging procedure without a trimming level, built on exact clusterwise least-squares
  fits of small random subsamples drawn sequentially from the units not yet flagged. The small
  exact problems are solved by a depth-first branch and bound, which the manuscript compares with
  mixed-integer programming solvers. Followed by the recovery step, ESF was about as accurate as
  the better of two trimming levels chosen after the fact when up to a fifth of the data were
  outliers.

The simulation study covers 7,750 data sets and runs TCLUST-REG from the FSDA toolbox with a
long random search at four levels and with equal weights, each followed by reweighting, and
also, with a shorter search, with adaptive second-level trimming and with a level chosen from
its monitoring path. It reports where the proposals fail as
well as where they succeed: the generous level is more robust than ESF beyond a quarter of
outliers, no fixed level is best at both low and high contamination, and the recovery step can
accept a narrow band of uniform noise as a group. One of the three data sets, taxi fares from
New York's JFK airport, records the tariff of every trip, so the true group of every unit is
known.

The code, the raw results and the dated analysis plans are public. The manuscript is not under
consideration elsewhere. I declare no competing interests. The use of a large language model is
described in the Declarations of the manuscript.

Yours sincerely,

Samir Orujov
School of Business, ADA University, Baku, Azerbaijan
sorujov@ada.edu.az, ORCID 0009-0004-9708-2109
