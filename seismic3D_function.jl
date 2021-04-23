

function implement_2D_forward(dt,dx,dz,nt,nx,nz,
    X,Z,
    r1,r3,
    Rm,
    s_s1,s_s3,s_src1,s_src3,s_source_type,
    r1t,r3t,
    s_s1t,s_s3t,
    lp,nPML,Rc,
    C,
    plot_interval,
    wavefield_interval,
    p3)

    global path,v1,v3,R1,R3,P

    # initialize seismograms
    R1=zeros(Float32,nt,length(r3));
    R3=copy(R1);
    P=copy(R1);
    data=zeros(nt,length(r3));

    ##
    @time begin
        if msot==1
            global s1,s3,s1t,s3t,src1,src3,source_type,v1,v3,R1,R3,P,path,data,
            path_pic,path_model,path_wavefield,path_rec
            # path for this source
            path=p3;
            if isdir(string(path))==0
                mkdir(string(path));
            end;
            path_pic=string(path,"/pic");
            path_model=string(path,"/model");
            path_wavefield=string(path,"/wavefield");
            path_rec=string(path,"/rec");

            # source locations
            s1=s_s1;
            s3=s_s3;

            s1t=s_s1t;
            s3t=s_s3t;

            src1=s_src1;
            src3=s_src3;
            source_type=s_source_type;

            # pass parameters to solver
            v1,v3,R1,R3,P=mono_2D(dt,dx,dz,nt,
            nx,nz,X,Z,r1,r3,s1,s3,src1,src3,source_type,
            r1t,r3t,
            Rm,
            s1t,s3t,
            lp,nPML,Rc,
            C,
            plot_interval,
            wavefield_interval,
            path,
            path_pic,
            path_model,
            path_wavefield,
            path_rec);


        else
            for source_code=1:length(s_s3)
                global s1,s3,s1t,s3t,src1,src3,source_type,v1,v3,R1,R3,P,path,data,
                path_pic,path_model,path_wavefield,path_rec
                # source locations
                s1=s_s1[source_code];
                s3=s_s3[source_code];

                # path for this source
                path=string(p3,"/source_code_",
                (source_code));

                path_pic=string(path,"/pic");
                path_model=string(path,"/model");
                path_wavefield=string(path,"/wavefield");
                path_rec=string(path,"/rec");

                s1=s_s1[source_code];
                s3=s_s3[source_code];

                s1t=s_s1t[source_code];
                s3t=s_s3t[source_code];

                src1=s_src1[:,source_code];
                src3=s_src3[:,source_code];
                source_type=string(s_source_type[source_code]);


                # pass parameters to solver
                v1,v3,R1,R3,P=mono_2D(dt,dx,dz,nt,
                nx,nz,X,Z,r1,r3,s1,s3,src1,src3,source_type,
                r1t,r3t,
                Rm,
                s1t,s3t,
                lp,nPML,Rc,
                C,
                plot_interval,
                wavefield_interval,
                path,
                path_pic,
                path_model,
                path_wavefield,
                path_rec);
            end
        end
    end
end

function meshgrid(x,y)
    x2=zeros(length(x),length(y));
    y2=x2;
    x2=repeat(x,1,length(y));
    y2=repeat(reshape(y,1,length(y)),length(x),1);
    return x2,y2
end

function write2mat(path,var)
    file=matopen(path,"w");
    write(file,"data",data);
    close(file);
    return nothing
end

function readmat(path,var)
    file=matopen(path);
    tt=read(file,var);
    close(file)
    return tt
end

function rickerWave(freq,dt,ns,M)
    ## calculate scale
    E=10 .^(5.24+1.44 .*M);
    s=sqrt(E.*freq/.299);

    t=dt:dt:dt*ns;
    t0=1 ./freq;
    t=t .-t0;
    ricker=s .*(1 .-2*pi^2*freq .^2*t .^2).*exp.(-pi^2*freq^2 .*t .^2);
    ricker=ricker;
    ricker=Float32.(ricker);
    return ricker
end
##
@parallel function compute_sigma(dt,dx,dy,dz,inv_Qa,
    C11,C12,C13,C14,C15,C16,
    C22,C23,C24,C25,C26,
    C33,C34,C35,C36,
    C44,C45,C46,
    C55,C56,
    C66,
    beta,
    v1,v1_2_2_end,v1_3_2_end,
    v2,v2_1_2_end,v2_3_2_end,
    v3,v3_1_2_end,v3_2_2_end,
    sigmas11,sigmas22,sigmas33,sigmas23,sigmas13,sigmas12,p,
    ax,ax2,ax3,ax4,ax5,ax6,ax7,
    Ax,Ax2,Ax3,Ax4,Ax5,Ax6,Ax7,
    ax_dt,ax2_dt,ax3_dt,ax4_dt,ax5_dt,ax6_dt,ax7_dt)

    @inn(Ax)=@inn(ax);
    @inn(Ax2)=@inn(ax2);
    @inn(Ax3)=@inn(ax3);
    @inn(Ax4)=@inn(ax4);
    @inn(Ax5)=@inn(ax5);
    @inn(Ax6)=@inn(ax6);
    @inn(Ax7)=@inn(ax7);

    @inn(ax)=(2*@all(C11)-@all(C12)-@all(C13)) .*@d_xi(v1)/dx+
    (2*@all(C16)-@all(C26)-@all(C36)) .*@d_yi(v1_2_2_end)/dy+
    (2*@all(C15)-@all(C25)-@all(C35)) .*@d_zi(v1_3_2_end)/dz+
    (2*@all(C16)-@all(C26)-@all(C36)) .*@d_xi(v2_1_2_end)/dx+
    (2*@all(C12)-@all(C22)-@all(C23)) .*@d_yi(v2)/dy+
    (2*@all(C14)-@all(C24)-@all(C34)) .*@d_zi(v2_3_2_end)/dz+
    (2*@all(C15)-@all(C25)-@all(C35)) .*@d_xi(v3_1_2_end)/dx+
    (2*@all(C14)-@all(C24)-@all(C34)) .*@d_yi(v3_2_2_end)/dy+
    (2*@all(C13)-@all(C23)-@all(C33)) .*@d_zi(v3)/dz;

    @inn(ax2)=(-@all(C11)+2*@all(C12)-@all(C13)) .*@d_xi(v1)/dx+
    (-@all(C16)+2*@all(C26)-@all(C36)) .*@d_yi(v1_2_2_end)/dy+
    (-@all(C15)+2*@all(C25)-@all(C35)) .*@d_zi(v1_3_2_end)/dz+
    (-@all(C16)+2*@all(C26)-@all(C36)) .*@d_xi(v2_1_2_end)/dx+
    (-@all(C12)+2*@all(C22)-@all(C23)) .*@d_yi(v2)/dy+
    (-@all(C14)+2*@all(C24)-@all(C34)) .*@d_zi(v2_3_2_end)/dz+
    (-@all(C15)+2*@all(C25)-@all(C35)) .*@d_xi(v3_1_2_end)/dx+
    (-@all(C14)+2*@all(C24)-@all(C34)) .*@d_yi(v3_2_2_end)/dy+
    (-@all(C13)+2*@all(C23)-@all(C33)) .*@d_zi(v3)/dz;

    @inn(ax3)=(-1*@all(C11)-@all(C12)+2*@all(C13)) .*@d_xi(v1)/dx+
    (-1*@all(C16)-@all(C26)+2*@all(C36)) .*@d_yi(v1_2_2_end)/dy+
    (-1*@all(C15)-@all(C25)+2*@all(C35)) .*@d_zi(v1_3_2_end)/dz+
    (-1*@all(C16)-@all(C26)+2*@all(C36)) .*@d_xi(v2_1_2_end)/dx+
    (-1*@all(C12)-@all(C22)+2*@all(C23)) .*@d_yi(v2)/dy+
    (-1*@all(C14)-@all(C24)+2*@all(C34)) .*@d_zi(v2_3_2_end)/dz+
    (-1*@all(C15)-@all(C25)+2*@all(C35)) .*@d_xi(v3_1_2_end)/dx+
    (-1*@all(C14)-@all(C24)+2*@all(C34)) .*@d_yi(v3_2_2_end)/dy+
    (-1*@all(C13)-@all(C23)+2*@all(C33)) .*@d_zi(v3)/dz;

    @inn(ax4)=@all(C16).*(@d_xi(v1)/dx)+
    @all(C56).*(@d_xi(v3_1_2_end)/dx)+
    @all(C66).*(@d_xi(v2_1_2_end)/dx)+
    @all(C26).*(@d_yi(v2)/dy)+
    @all(C46).*(@d_yi(v3_2_2_end)/dy)+
    @all(C66).*(@d_yi(v1_2_2_end)/dy)+
    @all(C36).*(@d_zi(v3)/dz)+
    @all(C46).*(@d_zi(v2_3_2_end)/dz)+
    @all(C56).*(@d_zi(v1_3_2_end)/dz);

    @inn(ax5)=@all(C15) .*(@d_xi(v1)/dx)+
    @all(C56) .*(@d_xi(v2_1_2_end)/dx)+
    @all(C55) .*(@d_xi(v3_1_2_end)/dx)+
    @all(C25) .*(@d_yi(v2)/dy)+
    @all(C45) .*(@d_yi(v3_2_2_end)/dy)+
    @all(C56) .*(@d_yi(v1_2_2_end)/dy)+
    @all(C35) .*(@d_zi(v3)/dz)+
    @all(C45) .*(@d_zi(v2_3_2_end)/dz)+
    @all(C55) .*(@d_zi(v1_3_2_end)/dz);

    @inn(ax6)=@all(C14).*(@d_xi(v1)/dx)+
    @all(C46).*(@d_xi(v2_1_2_end)/dx)+
    @all(C45).*(@d_xi(v3_1_2_end)/dx)+
    @all(C24).*(@d_yi(v2)/dy)+
    @all(C46).*(@d_yi(v1_2_2_end)/dy)+
    @all(C44).*(@d_yi(v3_2_2_end)/dy)+
    @all(C34).*(@d_zi(v3)/dz)+
    @all(C45).*(@d_zi(v1_3_2_end)/dz)+
    @all(C44).*(@d_zi(v2_3_2_end)/dz);

    @inn(ax7)=(@all(C11)+@all(C12)+@all(C13)) .*@d_xi(v1)/dx+
    (@all(C16)+@all(C26)+@all(C36)) .*@d_yi(v1_2_2_end)/dy+
    (@all(C15)+@all(C25)+@all(C35)) .*@d_zi(v1_3_2_end)/dz+
    (@all(C16)+@all(C26)+@all(C36)) .*@d_xi(v2_1_2_end)/dx+
    (@all(C12)+@all(C22)+@all(C23)) .*@d_yi(v2)/dy+
    (@all(C14)+@all(C24)+@all(C34)) .*@d_zi(v2_3_2_end)/dz+
    (@all(C15)+@all(C25)+@all(C35)) .*@d_xi(v3_1_2_end)/dx+
    (@all(C14)+@all(C24)+@all(C34)) .*@d_yi(v3_2_2_end)/dy+
    (@all(C13)+@all(C23)+@all(C33)) .*@d_zi(v3)/dz;

    @inn(ax_dt)=(@inn(ax)-@inn(Ax))/dt;
    @inn(ax2_dt)=(@inn(ax2)-@inn(Ax2))/dt;
    @inn(ax3_dt)=(@inn(ax3)-@inn(Ax3))/dt;
    @inn(ax4_dt)=(@inn(ax4)-@inn(Ax4))/dt;
    @inn(ax5_dt)=(@inn(ax5)-@inn(Ax5))/dt;
    @inn(ax6_dt)=(@inn(ax6)-@inn(Ax6))/dt;
    @inn(ax7_dt)=(@inn(ax7)-@inn(Ax7))/dt;

    @inn(sigmas11)=1/3*dt*(
    @inn(ax)+@all(inv_Qa) .*@inn(ax_dt))+
    @inn(sigmas11)-
    dt*@all(beta).*@inn(sigmas11);

    @inn(sigmas22)=1/3*dt*(
    @inn(ax2)+@all(inv_Qa) .*@inn(ax2_dt))+
    @inn(sigmas22)-
    dt*@all(beta).*@inn(sigmas22);

    @inn(sigmas33)=1/3*dt*(
    @inn(ax3)+@all(inv_Qa) .*@inn(ax3_dt))+
    @inn(sigmas33)-
    dt*@all(beta).*@inn(sigmas33);

    @inn(sigmas12)=dt*(
    @inn(ax4)+@all(inv_Qa) .*@inn(ax4_dt))+
    @inn(sigmas12)-
    dt*@all(beta).*@inn(sigmas12);

    @inn(sigmas13)=dt*(
    @inn(ax5)+@all(inv_Qa) .*@inn(ax5_dt))+
    @inn(sigmas13)-
    dt*@all(beta).*@inn(sigmas13);

    @inn(sigmas23)=dt*(
    @inn(ax6)+@all(inv_Qa) .*@inn(ax6_dt))+
    @inn(sigmas23)-
    dt*@all(beta).*@inn(sigmas23);

    @inn(p)=-1/3*dt*(
    @inn(ax7)+@all(inv_Qa) .*@inn(ax7_dt))+
    @inn(p)-
    dt*@all(beta).*@inn(p);

    return nothing
end

@parallel_indices (iy,iz) function x_2_end(in,out)
out[:,iy,iz]=in[2:end,iy,iz];
return nothing
end

@parallel_indices (ix,iz) function y_2_end(in,out)
out[ix,:,iz]=in[ix,2:end,iz];
return nothing
end

@parallel_indices (ix,iy) function z_2_end(in,out)
out[ix,iy,:]=in[ix,iy,2:end];
return nothing
end
##
@parallel function compute_v(dt,dx,dy,dz,rho,beta,
    v1,v2,v3,
    sigmas11_minus_p_1_2_end,sigmas12,sigmas13,
    sigmas22_minus_p_2_2_end,sigmas23,
    sigmas33_minus_p_3_2_end)

    @inn(v1)=dt./@all(rho) .*(@d_xi(sigmas11_minus_p_1_2_end)/dx+
    @d_yi(sigmas12)/dy+
    @d_zi(sigmas13)/dz)+
    @inn(v1)-
    dt*@all(beta) .*@inn(v1);

    @inn(v2)=dt./@all(rho) .*(@d_xi(sigmas12)/dx+
    @d_yi(sigmas22_minus_p_2_2_end)/dy+
    @d_zi(sigmas23)/dz)+
    @inn(v2)-
    dt*@all(beta) .*@inn(v2);

    @inn(v3)=dt./@all(rho) .*(@d_xi(sigmas13)/dx+
    @d_yi(sigmas23)/dy+
    @d_zi(sigmas33_minus_p_3_2_end)/dz)+
    @inn(v3)-
    dt*@all(beta) .*@inn(v3);
    return nothing
end

@parallel function minus(a,b,c)
    @all(c)=@all(a)-@all(b);
    return nothing
end

@timeit ti "tric_3D" function tric_3D(dt,dx,dy,dz,nt,
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

global data

d0=Dates.now();
# source number
ns=length(s3);

# create main folder
if isdir(path)==0
    mkdir(path);
end

# create folder for picture
n_picture=1;
n_wavefield=1;
if path_pic!=nothing
    if isdir(path_pic)==0
        mkdir(path_pic);
    end
    # initialize pvd
    pvd=paraview_collection(string(path,"/time_info"));
end

# create folder for model
if path_model!=nothing
    if isdir(path_model)==0
        mkdir(path_model)
    end
    vtkfile = vtk_grid(string(path_model,"/C33"),X,Y,Z);
    vtkfile["C33"]=C.C33;
    vtk_save(vtkfile);
    CSV.write(string(path_model,"/receiver location.csv"),DataFrame([r1t' r2t' r3t']));
    CSV.write(string(path_model,"/source location.csv"),DataFrame([s1t' s2t' s3t']));
end

# create folder for wavefield
if path_wavefield!=nothing
    if isdir(path_wavefield)==0
        mkdir(path_wavefield)
    end
end


# create folder for rec
if path_rec!=nothing
    if isdir(path_rec)==0
        mkdir(path_rec)
    end
end

# PML
vmax=sqrt.((C.C33) ./C.rho);
beta0=(ones(nx,ny,nz) .*vmax .*(nPML+1) .*log(1/Rc)/2/lp/dx);
beta1=(@zeros(nx,ny,nz));
beta2=copy(beta1);
beta3=copy(beta1);
tt=(1:lp)/lp;


# PML coefficient
beta1=@zeros(nx,ny,nz);
tt=copy(beta1);
tt[2:lp+1,:,:]=repeat(reshape((abs.((1:lp) .-lp .-1) ./lp) .^nPML,lp,1,1),1,ny,nz);
tt[nx-lp:nx-1,:,:]=repeat(reshape(((abs.(nx .-lp .-(nx-lp+1:nx))) ./lp) .^nPML,lp,1,1),1,ny,nz);
beta1=vmax*(nPML+1)*log(1/Rc)/2/lp/dx.*tt;

beta2=zeros(nx,ny,nz);
tt=copy(beta2);
tt[:,2:lp+1,:]=repeat(reshape((abs.((1:lp) .-lp .-1) ./lp) .^nPML,1,lp,1),nx,1,nz);
tt[:,ny-lp:ny-1,:]=repeat(reshape(((abs.(ny .-lp .-(ny-lp+1:ny))) ./lp) .^nPML,1,lp,1),nx,1,nz);
beta2=vmax*(nPML+1)*log(1/Rc)/2/lp/dy.*tt;

beta3=zeros(nx,ny,nz);
tt=copy(beta3);
tt[:,:,2:lp+1]=repeat(reshape((abs.((1:lp) .-lp .-1) ./lp) .^nPML,1,1,lp),nx,ny,1);
tt[:,:,nz-lp:nz-1]=repeat(reshape(((abs.(nz .-lp .-(nz-lp+1:nz))) ./lp) .^nPML,1,1,lp),nx,ny,1);
beta3=vmax*(nPML+1)*log(1/Rc)/2/lp/dz.*tt;

beta1[1,:,:]=beta1[2,:,:];
beta1[end,:,:]=beta1[end-1,:,:];

beta2[:,1,:]=beta2[:,2,:];
beta2[:,end,:]=beta2[:,end-1,:];

beta3[:,:,1]=beta3[:,:,2];
beta3[:,:,end]=beta3[:,:,end-1];

# 3D PML coefficient
IND=unique(findall(x->x!=0,beta1.*beta2.*beta3));
IND2=unique(findall(x->x==x,beta1.*beta2+beta2.*beta3+beta3.*beta1));
IND3=setdiff(IND2,IND);
beta=beta1+beta2+beta3;
beta[IND]=beta[IND]/3;
beta[IND3]=beta[IND3]/2;

vmax=beta01=beta02=beta03=tt=beta1=beta2=beta3=IND=IND2=IND3=nothing;

# receiver configuration
R1=@zeros(nt,length(r3));
R2=copy(R1);
R3=copy(R1);
P=@zeros(nt,length(r3));

# wave vector
v1=@zeros(nx,ny,nz);
v2=copy(v1);
v3=copy(v1);

sigmas11=copy(v1);
sigmas22=copy(v1);
sigmas33=copy(v1);
sigmas23=copy(v1);
sigmas13=copy(v1);
sigmas12=copy(v1);
p=copy(v1);

ax=copy(v1);
ax2=copy(v1);
ax3=copy(v1);
ax4=copy(v1);
ax5=copy(v1);
ax6=copy(v1);
ax7=copy(v1);
Ax=copy(v1);
Ax2=copy(v1);
Ax3=copy(v1);
Ax4=copy(v1);
Ax5=copy(v1);
Ax6=copy(v1);
Ax7=copy(v1);
ax_dt=copy(v1);
ax2_dt=copy(v1);
ax3_dt=copy(v1);
ax4_dt=copy(v1);
ax5_dt=copy(v1);
ax6_dt=copy(v1);
ax7_dt=copy(v1);

l=1;
# save wavefield
if path_wavefield!=nothing && wavefield_interval!=0
    if mod(l,wavefield_interval)==0
        data=zeros(nx,ny,nz);
        write2mat(string(path_wavefield,"/v1_",n_wavefield,".mat"),data);
        data=zeros(nx,ny,nz);
        write2mat(string(path_wavefield,"/v3_",n_wavefield,".mat"),data);
        data=zeros(nx,ny,nz);
        write2mat(string(path_wavefield,"/sigmas11_",n_wavefield,".mat"),data);
        data=zeros(nx,ny,nz);
        write2mat(string(path_wavefield,"/sigmas33_",n_wavefield,".mat"),data);
        data=zeros(nx,ny,nz);
        write2mat(string(path_wavefield,"/sigmas13_",n_wavefield,".mat"),data);
        data=zeros(nx,ny,nz);
        write2mat(string(path_wavefield,"/p_",n_wavefield,".mat"),data);
        n_wavefield=n_wavefield+1;
    end
end
#
v1_2_2_end=@zeros(nx,ny-1,nz);
v1_3_2_end=@zeros(nx,ny,nz-1);
v2_1_2_end=@zeros(nx-1,ny,nz);
v2_3_2_end=@zeros(nx,ny,nz-1);
v3_1_2_end=@zeros(nx-1,ny,nz);
v3_2_2_end=@zeros(nx,ny-1,nz);

sigmas11_minus_p_1_2_end=@zeros(nx-1,ny,nz);
sigmas22_minus_p_2_2_end=@zeros(nx,ny-1,nz);
sigmas33_minus_p_3_2_end=@zeros(nx,ny,nz-1);
sigmas11_minus_p=@zeros(nx,ny,nz);
sigmas22_minus_p=@zeros(nx,ny,nz);
sigmas33_minus_p=@zeros(nx,ny,nz);
pro_bar=Progress(nt,1,"forward_simulation...",50);
for l=1:nt-1
    @timeit ti "shift coordinate" @parallel (2:nx-1,2:nz-1) y_2_end(v1,v1_2_2_end);
    @timeit ti "shift coordinate" @parallel (2:nx-1,2:ny-1) z_2_end(v1,v1_3_2_end);

    @timeit ti "shift coordinate" @parallel (2:ny-1,2:nz-1) x_2_end(v2,v2_1_2_end);
    @timeit ti "shift coordinate" @parallel (2:nx-1,2:ny-1) z_2_end(v2,v2_3_2_end);

    @timeit ti "shift coordinate" @parallel (2:ny-1,2:nz-1) x_2_end(v3,v3_1_2_end);
    @timeit ti "shift coordinate" @parallel (2:nx-1,2:nz-1) y_2_end(v3,v3_2_2_end);

    @timeit ti "compute_sigma" @parallel compute_sigma(dt,dx,dy,dz,C.inv_Qa,
        C.C11,C.C12,C.C13,C.C14,C.C15,C.C16,
        C.C22,C.C23,C.C24,C.C25,C.C26,
        C.C33,C.C34,C.C35,C.C36,
        C.C44,C.C45,C.C46,
        C.C55,C.C56,
        C.C66,
        beta,
        v1,v1_2_2_end,v1_3_2_end,
        v2,v2_1_2_end,v2_3_2_end,
        v3,v3_1_2_end,v3_2_2_end,
        sigmas11,sigmas22,sigmas33,sigmas23,sigmas13,sigmas12,p,
        ax,ax2,ax3,ax4,ax5,ax6,ax7,
        Ax,Ax2,Ax3,Ax4,Ax5,Ax6,Ax7,
        ax_dt,ax2_dt,ax3_dt,ax4_dt,ax5_dt,ax6_dt,ax7_dt);

    @timeit ti "minus" @parallel minus(sigmas11,p,sigmas11_minus_p);
    @timeit ti "minus" @parallel minus(sigmas22,p,sigmas22_minus_p);
    @timeit ti "minus" @parallel minus(sigmas33,p,sigmas33_minus_p);

    @timeit ti "shift coordinate" @parallel (2:ny-1,2:nz-1) x_2_end(sigmas11_minus_p,sigmas11_minus_p_1_2_end);
    @timeit ti "shift coordinate" @parallel (2:nx-1,2:nz-1) y_2_end(sigmas22_minus_p,sigmas22_minus_p_2_2_end);
    @timeit ti "shift coordinate" @parallel (2:nx-1,2:ny-1) z_2_end(sigmas33_minus_p,sigmas33_minus_p_3_2_end);

    @timeit ti "compute_v" @parallel compute_v(dt,dx,dy,dz,C.rho,beta,
        v1,v2,v3,
        sigmas11_minus_p_1_2_end,sigmas12,sigmas13,
        sigmas22_minus_p_2_2_end,sigmas23,
        sigmas33_minus_p_3_2_end);

    @timeit ti "source" if source_type=="D"
    if ns==1
        v1[CartesianIndex.(s1,s2,s3)]=v1[CartesianIndex.(s1,s2,s3)]+1 ./C.rho[CartesianIndex.(s1,s2,s3)] .*src1[l];
        v2[CartesianIndex.(s1,s2,s3)]=v2[CartesianIndex.(s1,s2,s3)]+1 ./C.rho[CartesianIndex.(s1,s2,s3)] .*src2[l];
        v3[CartesianIndex.(s1,s2,s3)]=v3[CartesianIndex.(s1,s2,s3)]+1 ./C.rho[CartesianIndex.(s1,s2,s3)] .*src3[l];
    else
        v1[CartesianIndex.(s1,s2,s3)]=v1[CartesianIndex.(s1,s2,s3)]+1 ./C.rho[CartesianIndex.(s1,s2,s3)] .*src1[l,:]';
        v2[CartesianIndex.(s1,s2,s3)]=v2[CartesianIndex.(s1,s2,s3)]+1 ./C.rho[CartesianIndex.(s1,s2,s3)] .*src2[l,:]';
        v3[CartesianIndex.(s1,s2,s3)]=v3[CartesianIndex.(s1,s2,s3)]+1 ./C.rho[CartesianIndex.(s1,s2,s3)] .*src3[l,:]';
    end
end

@timeit ti "source" if source_type=="P"
if ns==1
    p[CartesianIndex.(s1,s2,s3)]=p[CartesianIndex.(s1,s2,s3)]+src3[l];
else
    p[CartesianIndex.(s1,s2,s3)]=p[CartesianIndex.(s1,s2,s3)]+src3[l,:]';
end
end

# assign recordings
@timeit ti "receiver" R1[l+1,:]=reshape(v1[CartesianIndex.(r1,r2,r3)],length(r3),);
@timeit ti "receiver" R2[l+1,:]=reshape(v2[CartesianIndex.(r1,r2,r3)],length(r3),);
@timeit ti "receiver" R3[l+1,:]=reshape(v3[CartesianIndex.(r1,r2,r3)],length(r3),);
@timeit ti "receiver" P[l+1,:]=reshape(p[CartesianIndex.(r1,r2,r3)],length(r3),);
# save wavefield
if path_wavefield!=nothing && wavefield_interval!=0
    if mod(l,wavefield_interval)==0
        data=v1;
        write2mat(string(path_wavefield,"/v1_",n_wavefield,".mat"),data);
        data=v3;
        write2mat(string(path_wavefield,"/v3_",n_wavefield,".mat"),data);
        data=sigmas11;
        write2mat(string(path_wavefield,"/sigmas11_",n_wavefield,".mat"),data);
        data=sigmas33;
        write2mat(string(path_wavefield,"/sigmas33_",n_wavefield,".mat"),data);
        data=sigmas13;
        write2mat(string(path_wavefield,"/sigmas13_",n_wavefield,".mat"),data);
        data=p;
        write2mat(string(path_wavefield,"/p_",n_wavefield,".mat"),data);
        n_wavefield=n_wavefield+1;
    end
end

# plot
if path_pic!=nothing && plot_interval!=0
    if mod(l,plot_interval)==0 || l==nt-1
        vtkfile = vtk_grid(string(path_pic,"/wavefield_pic_",n_picture),X,Y,Z);
        vtkfile["v1"]=v1;
        vtkfile["v2"]=v2;
        vtkfile["v3"]=v3;
        vtkfile["p"]=p;
        vtkfile["C33"]=C.C33;
        vtkfile["C11"]=C.C11;
        pvd[dt*(l+1)]=vtkfile;
        n_picture=n_picture+1;
    end
end

next!(pro_bar);
end

R1=R1 .*Rm[:,:,1];
R2=R2 .*Rm[:,:,2];
R3=R3 .*Rm[:,:,2];
P=P .*Rm[:,:,3];

data=R1;
write2mat(string(path_rec,"/rec_1.mat"),data);
data=R2;
write2mat(string(path_rec,"/rec_2.mat"),data);
data=R3;
write2mat(string(path_rec,"/rec_3.mat"),data);
data=P;
write2mat(string(path_rec,"/rec_p.mat"),data);

if path_pic!=nothing && plot_interval!=0
    vtk_save(pvd);
end

return v1,v2,v3,R1,R2,R3,P
end
