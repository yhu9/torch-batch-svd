import torch
from torch_batch_svd import svd


def bench_speed(N, H, W):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.manual_seed(0)
    a = torch.randn(N, H, W).cuda()
    b = a.clone().cuda()
    torch.cuda.synchronize()

    start.record()
    for i in range(10):
        U, S, V = svd(a)
    end.record()
    torch.cuda.synchronize()
    t1 = start.elapsed_time(end) / 10
    print("Perform batched SVD on a {}x{}x{} matrix: {} ms".format(N, H, W, t1))

    start.record()
    for i in range(10):
        U, S, V = torch.linalg.svd(b)
    end.record()
    torch.cuda.synchronize()
    t2 = start.elapsed_time(end) / 10
    print("Perform torch.svd on a {}x{}x{} matrix: {} ms".format(N, H, W, t2))
    return t1, t2

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    times1 = []
    times2 = []
    xval = [i * 1000 for i in range(1,300)]
    for i in range(1,300):
        n = i * 1000
        t1,t2 = bench_speed(n, 3, 3)
        times1.append(t1)
        times2.append(t2)
    
    plt.plot(xval,times1,label='batch_svd')
    plt.plot(xval,times2,label='torch.svd')
    plt.xlabel('N 3x3 Matrices')
    plt.ylabel('Time in ms')
    plt.legend()
    plt.show()
    # plt.plot([i*10000 for i in range(1,30)]times)
        
    # bench_speed(20000, 9, 9)
