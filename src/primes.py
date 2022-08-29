"""
Harmonics can be seen as multiples of prime numbers.
But obviously, there are too many prime numbers to dedicate two dimensions to each.
Instead, we could detect, which prime numbers are reflected in the eigenvectors of a FFT spectrum.
  - What are common prime factors of the strong components of such an eigenvector?
  - Weight the corresponding eigenvalue, too.
  - eigenvalue/(prime factor)

Or simply use only the first (few) prime numbers
"""
upto = 50#96000

found = [2, 3]
current = 3
while current <= upto:
    current += 2
    current_is_prime = True
    for prime in found:
        if current % prime == 0:
            current_is_prime = False
            break
    if current_is_prime:
        found.append(current)

print(f"{len(found)} Primes: {found}")
        

