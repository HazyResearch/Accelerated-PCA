include("./Eigensolvers.jl")
using EigenSolvers
using PyPlot
srand(0)

function load(filename)
  f = open(filename)
  lines = readlines(f)
  close(f)

  nodes = parse(Int64, split(lines[3])[3])
  edges = parse(Int64, split(lines[3])[5])

  I = Array{Integer}(edges)
  J = Array{Integer}(edges)
  V = Array{Integer}(edges)

  for i = 1:edges
      I[i] = parse(Int64, split(lines[i + 4])[1])
      J[i] = parse(Int64, split(lines[i + 4])[2])
      V[i] = 1
  end
  
  # Id's aren't consecutive, map down
  dict = Dict{Int64, Int64}()
  count = 0
  for i = 1:edges
      if get(dict, I[i], 0) == 0
        count += 1
        dict[I[i]] = count
        I[i] = count
      else
        I[i] = get(dict, I[i], 0)
      end

      if get(dict, J[i], 0) == 0
        count += 1
        dict[J[i]] = count
        J[i] = count
      else
        J[i] = get(dict, J[i], 0)
      end
  end

  return sparse(I, J, V, nodes, nodes), nodes, edges, I, J
end

# A, nodes, edges, I, J = load("ca-GrQc.txt")
# A, nodes, edges, I, J = load("ca-CondMat.txt")
A, nodes, edges, I, J = load("ca-AstroPh.txt")
x0 = randn(nodes, 2)
println(mapslices(norm, x0, 1))
x0, _ = qr(x0)
println(mapslices(norm, x0, 1))
for i = 1:200
    x0 = A * x0
    println(mapslices(norm, x0, 1))
    x0, _ = qr(x0)
end

x0 = A * x0
Lambda = mapslices(norm, x0, 1)
u1 = x0[:, 1]
u1 /= norm(u1)

x0 = randn(nodes)
epoch = 100
beta = Lambda[2]^2/4
eig_gap = Lambda[1] - Lambda[2]

# figure(figsize=(6,5))
# res_pw = power(A, x0, 0, epoch, u1)
# println(res_pw)
# semilogy(res_pw, "b-",label="power")
# res_mpw = power(A, x0, beta, epoch, u1)
# println(res_mpw)
# semilogy(res_mpw, "r-",label="power+M")
# 
# num_trials = 1
# for s = [40000]
#     res_sgd = zeros(epoch+1, num_trials)
#     for i = 1:num_trials
#         res_sgd[:,i] = minibatch_sgd_m(I, J, x0,  0, epoch, u1, s,i)
#     end
#     semilogy(mean(res_sgd,2), label=@sprintf("mini-batch power(s=%d)", s),"b-.",)
#     res_msgd = zeros(epoch+1, num_trials)
#     for i = 1:num_trials
#         res_msgd[:,i] = minibatch_sgd_m(I, J, x0, beta, epoch, u1, s,i)
#     end
#     semilogy(mean(res_msgd,2), label=@sprintf("mini-batch power+M(s=%d)", s),"r-.",)
# end
# 
# 
# for s = [200000]
#     res_sgd = zeros(epoch+1, num_trials)
#     for i = 1:num_trials
#         res_sgd[:,i] = minibatch_sgd_m(I, J, x0, 0,  epoch, u1, s,i)
#     end
#     semilogy(mean(res_sgd ,2), label=@sprintf("mini-batch power(s=%d)", s),"b--",)
#     res_msgd = zeros(epoch+1, num_trials)
#     for i = 1:num_trials
#         res_msgd[:,i] = minibatch_sgd_m(I, J, x0, beta, epoch, u1, s,i)
#     end
#     semilogy(mean(res_msgd,2), label=@sprintf("mini-batch power+M(s=%d)", s),"r--",)
# end
# legend(loc="best",framealpha=0.7)
# xlabel("# Iteration", fontsize=20)
# ylabel(L"$1 - (w_t^Tu_1)^2$", fontsize=20)
# ylim([1e-9,1])
# grid("on")
# savefig("mini_power.pdf", bbox_inches="tight")

epoch = 100
figure(figsize=(6,5))
res_pw = power(A, x0, 0, epoch, u1)
println(res_pw)
semilogy(res_pw, "b-",label="power")
res_mpw = power(A, x0, beta, epoch, u1)
println(res_mpw)
semilogy(res_mpw, "r-",label="power+M")

m = 5
epoch = Int(epoch / m)
num_trials = 1
for s = [200000]
    res_sgd = zeros(epoch*m + 1, num_trials)
    res_msgd = zeros(epoch*m + 1, num_trials)
    for i = 1:num_trials
        res_sgd[:,i] = mini_batch_svrg_m(I, J, x0, 0, epoch,m, u1, s,i)
        println(res_sgd)
        res_msgd[:,i] = mini_batch_svrg_m(I, J, x0, beta, epoch,m, u1, s,i)
        println(res_msgd)
    end
    semilogy(mean(res_sgd,2), label=@sprintf("VR-power(s=%d)", s),"b--",)
    semilogy(mean(res_msgd,2), label=@sprintf("VR-power+M(s=%d)", s),"r--",)
end

# for s = [10000]
#     res_sgd = zeros(epoch*m + 1, num_trials)
#     res_msgd = zeros(epoch*m + 1, num_trials)
#     for i = 1:num_trials
#         res_sgd[:,i] = mini_batch_svrg_m(I, J, x0,0, epoch,m, u1, s,i)
#         res_msgd[:,i] = mini_batch_svrg_m(I, J, x0, beta, epoch,m, u1, s,i)
#     end
#     semilogy(mean(res_sgd,2), label=@sprintf("VR-power(s=%d)", s),"b-.")
#     semilogy(mean(res_msgd,2), label=@sprintf("VR-power+M(s=%d)", s),"r-.",)
# end
legend(loc="best",framealpha=0.7)
xlabel("# Iterations (K=20, T=10)", fontsize=20)
ylabel(L"$1 - (\tilde w_t^Tu_1)^2$", fontsize=20)
ylim([1e-15,1])
grid("on")
savefig("vr_power.pdf", bbox_inches="tight")
