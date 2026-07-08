
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuralnetwork import MLP

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  
model = MLP(3, [4, 4, 1])
print(model)
print("number of parameters:", len(model.parameters()))

lr = 0.05  
for step in range(100):
    
    ypred = [model(x) for x in xs]

    
    loss = sum((yp - yt) ** 2 for yp, yt in zip(ypred, ys))

   
    model.zero_grad()   
    loss.backward()

    # 4. update: nudge every parameter against its gradient
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 10 == 0:
        print(f"step {step:3d} | loss {loss.data:.4f}")

print("\nfinal predictions vs targets:")
for x, yt in zip(xs, ys):
    print(f"  pred {model(x).data:+.4f}   target {yt:+.1f}")
