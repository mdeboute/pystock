from pystock import Portfolio

portfolio = Portfolio.from_xlsx_file(r"D:\perso\pystock\data\etf.xlsx")
mcs = portfolio.monte_carlo_simulation()
pf = portfolio.get_pareto_front(mcs)
portfolio.plot_pareto_front(pf)