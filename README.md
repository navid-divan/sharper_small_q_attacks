# Efficient Small-q Attacks on Short Integer Solution
This work improves the classical small-q attack by [[DEP23]](https://link.springer.com/chapter/10.1007/978-3-031-38548-3_6). 
A sharper small-q attack is implemented in `efficient_smallqSIS.py`, and to compare it with classical one, test:
```
python3 benchmark_costs.py 256 257 falcon  # default: n=256 q=257 falcon 
python3 benchmark_costs.py 256 521 mitaka  4
python3 benchmark_costs.py 512 257 falcon  4     # b_step=4 is the safe default at n=512
python3 benchmark_costs.py 512 257 mitaka  6     # use b_step=6 if 4 OOMs
```
Also, we design further attack as the Small-q$`^∞`$, which is implemented in `linfty_smallqSIS.py`, and Dilithium-style ℓ∞ analogue $`ISIS^∞_{n,m,q,β∞}`$ replaces the Euclidean acceptance condition $`‖x‖ ≤ ν`$ of the original problem, and can test it by:

```
sage benchmark_linfty_attack.sage fast 5 # default is 3 trials
sage benchmark_linfty_attack.sage medium 3
```
To test these attacks, you need to install [Sage](https://doc.sagemath.org/html/en/installation/), along with the general [Sieve](https://github.com/fplll/g6k) kernel ([G6K](https://link.springer.com/chapter/10.1007/978-3-030-17656-3_25)).

For further information on this project, refer to [Classical Small-q](https://github.com/verdiverdiverdi/ISIS-small-q).
