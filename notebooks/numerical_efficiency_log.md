```
First method of computing efficiency $\frac{1 - \frac{r}{r_0}}{t}$ where $r$ is residual, $r_0$ is original residual,
and t is time
```
			residual	time	efficiency	log10_efficiency	preconditioning
adj			6.969512	0.016984	58.596485	1.775221	True
adj			25.716905	0.020999	46.782145	1.679266	False
tnc			69.103832	0.020986	45.397928	1.666499	True
tnc			69.103832	0.021979	43.345869	1.646853	False
l-bfgs-b	11.825107	0.023988	41.350097	1.626854	True
cg			10.894020	0.038027	26.100994	1.432985	False
minres		64.905333	0.039963	23.911242	1.396395	True
minres		64.905333	0.041962	22.772475	1.376074	False
cg			10.894020	0.060950	16.284546	1.237658	True
l-bfgs-b	11.825107	0.067975	14.592164	1.192906	False
gmres		0.841951	0.124929	7.999906	0.954238	True
gmres		0.841951	0.138870	7.196817	0.913645	False
bicgstab	39.515904	0.201928	4.818323	0.764798	True
cgs			83.665728	0.209841	4.492611	0.739779	True
bicgstab	39.515904	0.217801	4.467172	0.737763	False
cgs			83.665728	0.214075	4.403759	0.732696	False
lgmres		0.778212	0.227865	4.386226	0.731285	True
lgmres		0.778212	0.283725	3.522668	0.655395	False
newton-cg	0.010094	0.415739	2.405338	0.532160	True
newton-cg	0.010094	0.449588	2.224243	0.508428	False
lsmr		1.828582	2.154519	0.463560	0.165410	True
lsmr		1.828582	2.275862	0.438844	0.158014	False
lsqr		2.297405	5.170184	0.193113	0.076681	True
lsqr		2.297405	6.308716	0.158262	0.063807	False
lstsq		0.029957	30.434108	0.032857	0.014040	False
lstsq		27.291439	30.693768	0.031971	0.013668	True

lsmr, lsqr, lstsq shouldn't be used unless high accuracy required (lstsq). 
lsmr, lsqr have many parameters to tune, damp=10, atol=0.001, btol=0.001 seems
to work relatively well if damping is allowed. 

BICGSTAB and CGS disallowed for now as I cannot figure out how to work around
depreciation of scipy keyword arguments.  

Trust-region methods require Hessian (or its vector product) which is 
too much work for not enough gain currently. 