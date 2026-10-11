import wfdb


res=[str(i) for i in range(100,110)]
wfdb.dl_database(
    db_dir="mitdb",
    dl_dir="G:/project/dl_sem_pro/Wav-KAN/data",
    records="all"
)

