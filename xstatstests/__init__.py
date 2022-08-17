from .scipy import (
    ks_1samp_1d,
    ks_2samp_1d,
    anderson_ksamp,
    ttest_ind,
    ttest_rel,
    cramervonmises,
    cramervonmises_2samp,
    epps_singleton_2samp,
    mannwhitneyu,
    ranksums,
    kruskal,
    friedmanchisquare,
    brunnermunzel,
    ansari,
    bartlett,
    levene,
    fligner,
    median_test,
    mood,
    skewtest,
    kurtosistest,
    normaltest,
    jarque_bera,
    shapiro,
)
from .core import (
    ks_2samp_2d,
)

from . import _version

__version__ = _version.get_versions()["version"]
