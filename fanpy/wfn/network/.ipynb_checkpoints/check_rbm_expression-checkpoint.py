import numpy as np

nspin = 4  
nbath = 4

#a = [(random.random() * 0.04) - 0.02 for _ in range(nspin)]
#b = [(random.random() * 0.04) - 0.02 for _ in range(nbath)]
#w = [(random.random() * 0.04) - 0.02 for _ in range(nspin * nbath)]
#w = np.=np.array(w).reshape(nspin, nbath)

a =np.array([ 0.01950621, -0.00611817, -0.0170188 , -0.01222717]) 
b =np.array([-0.00800288, -0.01932492,  0.00167193,  0.01818047]) 
w =np.array([[ 0.01685886,  0.00056254, -0.01890393,  0.00315894],
       [ 0.00862846,  0.00300898,  0.01650293, -0.0117365 ],
       [ 0.01693179,  0.00805108, -0.00705406,  0.01365931],
       [-0.00980456,  0.01478897, -0.01725889,  0.00513892]]) 
sign_param =np.array([-0.01344758, -0.00381128, -0.01280593,  0.01465617,  0.00818625]) 

h = np.array([-1,  1, -1,  1])
occ_sds = np.array([[ 1., -1.,  1., -1.], 
       [-1.,  1.,  1., -1.], 
       [ 1., -1., -1.,  1.],  
       [-1.,  1., -1.,  1.]]) 

def numerator(n,h):
    for i in range(nspin):
        result = a[i] * n[i]
        for j in range(nbath):
            result += b[j] * h[j]
            result += w[i,j] * n[i] * h[j]
    return np.exp(result)

def sign(n):
    result = sign_param[-1]
    for i in range(nspin):
        result += sign_param[i] * n[i] 
    return np.tanh(result)

result = [numerator(n, h) for n in occ_sds]
signs = [sign(n) for n in occ_sds]
denominator = np.sum(result)
olp  = np.sqrt(result / denominator) * signs

print("numerator:", result)
print("denominator:", denominator)
print("olp:", olp)

    
