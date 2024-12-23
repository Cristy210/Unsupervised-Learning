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


# ╔═╡ Cell order:
# ╟─38def4ac-c0e5-11ef-34f2-0f088ddaacd4
# ╠═7b6bb3f6-aa61-495f-8c22-1bfd9253b316
# ╠═3f69f17b-e2b1-4b37-acdb-db326ed1ae25
# ╠═b606030d-f326-4ced-a440-8d7d4cad42b9
# ╠═c5043684-060d-485c-b820-821423f332e1
# ╠═d331160a-c448-417f-81cc-a5ac8e8bedcc
