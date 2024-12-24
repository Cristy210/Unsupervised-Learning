### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 7b6bb3f6-aa61-495f-8c22-1bfd9253b316
import Pkg; Pkg.activate(@__DIR__)


# ╔═╡ 3f69f17b-e2b1-4b37-acdb-db326ed1ae25
using CSV, DataFrames

# ╔═╡ 38def4ac-c0e5-11ef-34f2-0f088ddaacd4
html"""<style>
input[type*="range"] {
	width: calc(100% - 4rem);
}
main {
    max-width: 96%;
    margin-left: 0%;
    margin-right: 2% !important;
}
"""

# ╔═╡ b606030d-f326-4ced-a440-8d7d4cad42b9
Data_dir = joinpath(@__DIR__, "Datasets", "data.csv")

# ╔═╡ c5043684-060d-485c-b820-821423f332e1
Data_mat = CSV.read(Data_dir, DataFrame)

# ╔═╡ d331160a-c448-417f-81cc-a5ac8e8bedcc
patient_ids = Data_mat[:, :id]

# ╔═╡ c37182d1-0c03-4eb5-9211-7de2d8b2914f
labels = Data_mat[:, :diagnosis]

# ╔═╡ d4a669ca-afaa-43d4-95a1-a44e3711eff5
feature_names = names(Data_mat)[3:32]

# ╔═╡ e962591d-eb17-4edc-a72c-e5dfda06c8fc
features_matrix = Matrix(Data_mat[:, feature_names])'

# ╔═╡ e2b96d88-0d65-434b-8aee-6c154d772ac5


# ╔═╡ Cell order:
# ╟─38def4ac-c0e5-11ef-34f2-0f088ddaacd4
# ╠═7b6bb3f6-aa61-495f-8c22-1bfd9253b316
# ╠═3f69f17b-e2b1-4b37-acdb-db326ed1ae25
# ╠═b606030d-f326-4ced-a440-8d7d4cad42b9
# ╠═c5043684-060d-485c-b820-821423f332e1
# ╠═d331160a-c448-417f-81cc-a5ac8e8bedcc
# ╠═c37182d1-0c03-4eb5-9211-7de2d8b2914f
# ╠═d4a669ca-afaa-43d4-95a1-a44e3711eff5
# ╠═e962591d-eb17-4edc-a72c-e5dfda06c8fc
# ╠═e2b96d88-0d65-434b-8aee-6c154d772ac5
