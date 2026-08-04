import numpy as np
import pybamm
import pandas as pd
import matplotlib.pyplot as plt

params = pybamm.ParameterValues("Chen2020")

soc = np.linspace(0, 1, 501)
x_0, x_100, y_100, y_0 = pybamm.lithium_ion.get_min_max_stoichiometries(params)

x = x_0 + soc * (x_100 - x_0)
y = y_0 + soc * (y_100 - y_0)

sto = pybamm.InputParameter("sto", expected_size=soc.size)  # <-- key change

Un_expr = pybamm.FunctionParameter("Negative electrode OCP [V]", {"sto": sto})
Up_expr = pybamm.FunctionParameter("Positive electrode OCP [V]", {"sto": sto})

Un = np.asarray(params.evaluate(Un_expr, inputs={"sto": x})).reshape(-1)
Up = np.asarray(params.evaluate(Up_expr, inputs={"sto": y})).reshape(-1)

ocv = Up - Un

df = pd.DataFrame({
    "OCV [V]": ocv,
    "SOC [pu]": soc,
})


ax = df.plot(x="SOC [pu]", y="OCV [V]", legend=False)
ax.set_xlabel("SOC [pu]")
ax.set_ylabel("OCV [V]")
fig = ax.get_figure()
fig.savefig("hack\\model\\output\\parameters\\ocv_soc_Chen2020.svg", format="svg", bbox_inches="tight")
plt.close(fig) 

df.to_csv("hack\\model\\output\\parameters\\ocv_soc_Chen2020.csv")
