# %%
import pandas as pd
from plotly import express as px
# %%
df = pd.read_csv("hyperparameter_tuning__random_search__results.csv")
df["params_selection_rate"] = df["params_selection_size"] / df["params_population_size"]
# %%
px.parallel_coordinates(
    df,
    dimensions=[col for col in df.columns if col.startswith("params_")],
    color="value",
    color_continuous_scale=px.colors.sequential.Viridis_r,
    labels={col: col.replace("params_", "") for col in df.columns if col.startswith("params_")},
    title="Hyperparameter Search Parallel Coordinates",
    template="plotly_dark"
).show()
# %%
for param in [c for c in df.columns if c.startswith("params_")]:
    fig = px.scatter(
        df,
        x=param,
        y="value",
        trendline="lowess",
        title=f"Objective vs {param.replace('params_', '')}",
        labels={param: param.replace("params_", ""), "value": "Objective"},
        template="plotly_dark",
    )
    fig.show()
# %%
fig = px.density_heatmap(
    df,
    x="params_population_size",
    y="params_generations",
    z="value",
    histfunc="avg",
    color_continuous_scale="Viridis_r",
    title="Objective Heatmap: Population Size vs Generations",
    labels={
        "params_population_size": "Population Size",
        "params_generations": "Generations",
        "value": "Objective"
    },
    template="plotly_dark",
)
fig.show()
# %%
fig = px.density_heatmap(
    df,
    x="params_population_size",
    y="params_selection_size",
    z="value",
    histfunc="avg",
    color_continuous_scale="Viridis_r",
    title="Objective Heatmap: Population Size vs Generations",
    labels={
        "params_population_size": "Population Size",
        "params_selection_size": "Selection Size",
        "value": "Objective"
    },
    template="plotly_dark",
)
fig.show()
# %%
df
# %%
