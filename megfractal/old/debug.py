import megfractal.study as sub

S = sub.Study.from_pickle('/neurospin/meg/meg_tmp/ScaledTime_Dragana_2019/MEG/study_sensor_mf.pkl')

S.stat_subj('H', stat='median', seg=['p1'])
