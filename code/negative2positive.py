
def apply(df):
    min_cons = df["cons.conf.idx"].min()
    min_emp = df["emp.var.rate"].min()
    df["cons.conf.idx"] = df["cons.conf.idx"] + abs(min_cons)
    df["emp.var.rate"] = df["emp.var.rate"] + abs(min_emp)
