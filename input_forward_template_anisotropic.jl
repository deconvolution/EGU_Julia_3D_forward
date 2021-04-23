## import packages
using MAT,Plots,Dates,TimerOutputs,WriteVTK,DataFrames,CSV,ProgressMeter

const USE_GPU=false;  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA,Float64,3);
else
    @init_parallel_stencil(Threads,Float64,3);
end
include("./seismic3D_function.jl");
Threads.nthreads()
## timing
ti=TimerOutput();
## read stiffness and density
nx=300;
ny=300;
nz=300;
mutable struct C2
    C11
    C12
    C13
    C14
    C15
    C16
    C22
    C23
    C24
    C25
    C26
    C33
    C34
    C35
    C36
    C44
    C45
    C46
    C55
    C56
    C66
    inv_Qa
    rho
end
vp=@ones(nx,ny,nz)*2000;
vs=vp/sqrt(2);

lambda=vp .^2-2*vs .^2;
mu=vs .^2;

inv_Qa=@ones(nx,ny,nz)*.003;

C=C2(lambda+2*mu,
lambda,
lambda,
@ones(nx,ny,nz)*10^4,
@ones(nx,ny,nz)*10^4,
@ones(nx,ny,nz)*10^5,
lambda+2*mu,
lambda,
@ones(nx,ny,nz)*10^4,
@ones(nx,ny,nz)*10^4,
@zeros(nx,ny,nz),
lambda+2*mu,
@zeros(nx,ny,nz),
@zeros(nx,ny,nz),
@zeros(nx,ny,nz),
mu,
@zeros(nx,ny,nz),
@zeros(nx,ny,nz),
mu,
@zeros(nx,ny,nz),
mu,
inv_Qa,
@ones(nx,ny,nz)*1,
);
## define model parameters
dt=10.0^-3;
dx=10;
dy=10;
dz=10;
nt=2000;

X=(1:nx)*dx;
Y=(1:nz)*dz;
Z=(1:nz)*dz;

# PML layers
lp=15;

# PML coefficient, usually 2
nPML=2;

# Theoretical coefficient, more PML layers, less R
# Empirical values
# lp=[10,20,30,40]
# R=[.1,.01,.001,.0001]
Rc=.001;
## source
# source location
# source location grid
s_s1=zeros(Int32,1,1);
s_s2=copy(s_s1);
s_s3=copy(s_s1);
s_s1[:] .=50;
s_s2[:] .=50;
s_s3[:] .=50;

# source locations true
s_s1t=minimum(X) .+ dx .*s_s1;
s_s2t=minimum(Y) .+ dy .*s_s2;
s_s3t=dz .*s_s3;

# magnitude
M=2.7;
# source frequency [Hz]
freq=5;

# source signal
singles=rickerWave(freq,dt,nt,M);

# give source signal to x direction
s_src1=zeros(Float32,nt,1);

s_src2=copy(s_src1);
# give source signal to z direction
s_src3=copy(s_src1);

for i=1:length(s_s3)
s_src3[:,i]=singles;
end

# source type. 'D' for directional source. 'P' for P-source.
s_source_type="D"^length(s_s3);
## receiver
r1t=zeros(Float32,1,1);
r2t=copy(r1t);
r3t=copy(r1t);
r1=zeros(Int32,1,size(r1t,2));
r2=copy(r1);
r3=copy(r1);
r1[:] .=50;
r2[:] .=50;
r3[:] .=50;

r1t[:] =r1 .*dx;
r2t[:] =r2 .*dy;
r3t[:] =r3 .*dz;
## plot
# point interval in time steps, 0 = no plot
plot_interval=200;
# save wavefield
wavefield_interval=0;
## create folder for saving
p2= @__FILE__;
p3=chop(p2,head=0,tail=3);
if isdir(p3)==0
    mkdir(p3);
end
## mute some receiver components
Rm=ones(nt,length(r3),3);
Rm[:,:,3] .=0;
## initialize seismograms
R1=zeros(Float32,nt,length(r3));
R3=copy(R1);
P=copy(R1);

## implement solver
#for source_code=1:length(s_s3)
source_code=1;
    global s1,s3,s1t,s3t,src1,src3,source_type,v1,v3,R1,R3,P,path,data,
    path_pic,path_model,path_wavefield,path_rec
    # source locations
    s1=s_s1[source_code];
    s2=s_s2[source_code];
    s3=s_s3[source_code];

    # path for this source
    path=string(p3,"/source_code_",
    (source_code));

    path_pic=string(path,"/pic");
    path_model=string(path,"/model");
    path_wavefield=string(path,"/wavefield");
    path_rec=string(path,"/rec");

    s1=s_s1[source_code];
    s2=s_s2[source_code];
    s3=s_s3[source_code];

    s1t=s_s1t[source_code];
    s2t=s_s2t[source_code];
    s3t=s_s3t[source_code];

    src1=s_src1[:,source_code];
    src2=s_src2[:,source_code];
    src3=s_src3[:,source_code];
    source_type=string(s_source_type[source_code]);

    # pass parameters to solver
    v1,v2,v3,R1,R2,R3,P=tric_3D(dt,dx,dy,dz,nt,
    nx,ny,nz,X,Y,Z,r1,r2,r3,s1,s2,s3,src1,src2,src3,source_type,
    r1t,r2t,r3t,
    Rm,
    s1t,s2t,s3t,
    lp,nPML,Rc,
    C,
    plot_interval,
    wavefield_interval,
    path,
    path_pic,
    path_model,
    path_wavefield,
    path_rec);
#end
## plot seismograms
file=matopen(string(path_rec,"/rec_p.mat"));
tt=read(file,"data");
close(file);
ir=1;
plot(dt:dt:dt*nt,tt[:,ir])
