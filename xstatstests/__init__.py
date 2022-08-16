from .core import (
    ks_2samp_1d,
    ks_2samp_2d,
    anderson_ksamp,
    ttest_ind,
    ttest_rel,
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
)

from . import _version

__version__ = _version.get_versions()["version"]
