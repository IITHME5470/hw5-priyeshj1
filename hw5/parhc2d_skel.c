#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<mpi.h>

void grid(int nx, int nxglob, int istglob, int ienglob, double xstglob, double xenglob, double *x, double *dx)
{
  int i, iglob;
  
  //*dx = (xen-xst)/(double)(nx-1);              // why won't this work?
  *dx = (xenglob - xstglob)/(double)(nxglob-1);  // why will  this work?

  for(i=0; i<nx; i++)
  {
    iglob = istglob + i;
    x[i] = xstglob + (double)iglob * (*dx);
  }
}

void enforce_bcs(int nx, int ny, double *x, double *y, double **T)
{
  int i, j;

  // left and right ends
  for(j=0; j<ny; j++)
  {
    T[0][j] = 0.0;    T[nx-1][j] = 0.0;
  }

  // top and bottom ends
  for(i=0; i<nx; i++)
  {
    T[i][0] = 0.0;    T[i][ny-1] = 0.0;
  }
}

void set_initial_condition(int nx, int ny, double *x, double *y, double **T, double dx, double dy)
{
  int i, j;
  double del=1.0;

  for(i=0; i<nx; i++)
    for(j=0; j<ny; j++)
    {
        T[i][j] = 0.25 * (tanh((x[i]-0.4)/(del*dx)) - tanh((x[i]-0.6)/(del*dx))) 
                       * (tanh((y[j]-0.4)/(del*dy)) - tanh((y[j]-0.6)/(del*dy)));
    }

  enforce_bcs(nx,ny,x,y,T); //ensure BCs are satisfied at t = 0
}

void get_rhs(int nx, int nxglob, int ny, int nyglob, int istglob, int ienglob, int jstglob, int jenglob, double dx, double dy, double *xleftghost, double *xrightghost, double *ybotghost, double *ytopghost, double kdiff, double *x, double *y, double **T, double **rhs)
{
  int i, j;
  double dxsq = dx*dx, dysq = dy*dy;

  // interior points first
  for(i=1; i<nx-1; i++)
   for(j=1; j<ny-1; j++)
     rhs[i][j] = kdiff*(T[i+1][j]+T[i-1][j]-2.0*T[i][j])/dxsq +
           kdiff*(T[i][j+1]+T[i][j-1]-2.0*T[i][j])/dysq ;

  // left boundary
  i = 0;
  if(istglob==0)  //processors adjacent to the left end of the domain
    for(j=1; j<ny-1; j++)
      rhs[i][j] = 0.0;
  else
    for(j=1; j<ny-1; j++)
      rhs[i][j] = kdiff*(T[i+1][j]+xleftghost[j]-2.0*T[i][j])/dxsq +     // T[i-1][j] replaced with xleftghost
                  kdiff*(T[i][j+1]+T[i][j-1]-2.0*T[i][j])/dysq;
 
  // right boundary
  i = nx-1;
  if(ienglob==nxglob-1)  //processors adjacent to the right end of the domain
    ...;
  else
    ...;
    // T[i+1][j] replaced with xrightghost
 
  // bottom boundary
  j = 0;
  if(jstglob==0)  //processors adjacent to the bottom end of the domain
    ...;
  else
    ...;
     // T[i][j-1] replaced with ybotghost
 
  // top boundary
  j = ny-1;
    ....;

  // bot-left corner
  i = 0; j = 0;
  if(istglob==0 || jstglob==0)  //processors adjacent to the left or bottom ends of the domain
      rhs[i][j] = 0.0;
  else
      rhs[i][j] = kdiff*(T[i+1][j]+xleftghost[j]-2.0*T[i][j])/dxsq +   // T[i-1][j] replaced with xleftghost
                  kdiff*(T[i][j+1]+ybotghost[i]-2.0*T[i][j])/dysq;     // T[i][j-1] replaced with ybotghost
 
  // bot-right corner
  i = nx-1; j = 0;
    ....;
 
 
  // top-left corner
  i = 0; j = ny-1;
    ....;
 
  // top-right corner
  i = nx-1; j = ny-1;
    ....;
 
}

void halo_exchange_2d_x(int rank, int rank_x, int rank_y, int size, int px, int py, int nx, int ny, int nxglob, int nyglob, double *x, double *y, double **T, double *xleftghost, double *xrightghost, double *sendbuf_x, double *recvbuf_x)
{
  MPI_Status status;
  FILE* fid;
  char debugfname[100];
  int left_nb, right_nb, i, j;
 
  // set left neighbours 
    left_nb = ...;

  // set right neighbours 
    right_nb = ...;

  // ---send to left; recv from right---
  // pack send buffer
  for(...)
    sendbuf_x[j] = ...;
  // send and recv
  MPI_Recv(recvbuf_x, ......);
  MPI_Send(sendbuf_x, ......);
  // unpack recv buffer
  for(...)
    xrightghost[j] = ...;

  // ---send to right; recv from left---
  // pack send buffer
  for(....)
    sendbuf_x[j] = ...;
  // send and recv
  MPI_Recv(recvbuf_x, ......);
  MPI_Send(sendbuf_x, ......);
  // unpack recv buffer
  for(...)
    xleftghost[j] = ...;
}

void halo_exchange_2d_y(int rank, int rank_x, int rank_y, int size, int px, int py, int nx, int ny, int nxglob, int nyglob, double *x, double *y, double **T, double *ybotghost, double *ytopghost, double *sendbuf_y, double *recvbuf_y)
{
  MPI_Status status;
  FILE* fid;
  char debugfname[100];
  int bot_nb, top_nb, i, j;
 
  // set bot neighbours 
    bot_nb = ...;

  // set top neighbours 
    top_nb = ...;

  // ---send to bot; recv from top---
  // pack send buffer
  for(....)
    sendbuf_y[i] = ....;
  // send and recv
  MPI_Recv(recvbuf_y, ....);
  MPI_Send(sendbuf_y, ....);
  // unpack recv buffer
  for(....)
    ytopghost[i] = ...;

  // ---send to top; recv from bot---
  // pack send buffer
  for(....)
    sendbuf_y[i] = ....;
  // send and recv
  MPI_Recv(recvbuf_y, ....);
  MPI_Send(sendbuf_y, ....);
  // unpack recv buffer
  for(....)
    ytopghost[i] = ...;
}

void timestep_FwdEuler(int rank, int size, int rank_x, int rank_y, int px, int py, int nx, int nxglob, int ny, int nyglob, int istglob, int ienglob, int jstglob, int jenglob, double dt, double dx, double dy, double *xleftghost, double *xrightghost, double *ybotghost, double *ytopghost, double kdiff, double *x, double *y, double **T, double **rhs, double *sendbuf_x, double *recvbuf_x, double *sendbuf_y, double *recvbuf_y)
{

  int i,j;

  // communicate information to get xleftghost and xrightghost
  halo_exchange_2d_x(rank, rank_x, rank_y, size, px, py, nx, ny, nxglob, nyglob, x, y, T, xleftghost, xrightghost, sendbuf_x, recvbuf_x);
  halo_exchange_2d_y(rank, rank_x, rank_y, size, px, py, nx, ny, nxglob, nyglob, x, y, T, ybotghost,    ytopghost, sendbuf_y, recvbuf_y);

  get_rhs(nx,nxglob,ny,nyglob,istglob,ienglob,jstglob,jenglob,dx,dy,xleftghost,xrightghost,ybotghost,ytopghost,kdiff,x,y,T,rhs);

  // (Forward) Euler scheme
  for(i=0; i<nx; i++)
   for(j=0; j<ny; j++)
     T[i][j] = T[i][j] + dt*rhs[i][j];                           // update T^(it+1)[i]

  // set Dirichlet BCs
  enforce_bcs(nx,ny,x,y,T);

}

double get_error_norm_2d(int nx, int ny, double **arr1, double **arr2)
{
  double norm_diff = 0.0, local_diff;
  int i, j;

  for(i=0; i<nx; i++)
   for(j=0; j<ny; j++)
   {
     local_diff = arr1[i][j] - arr2[i][j];
     norm_diff += local_diff * local_diff;
   }
   norm_diff = sqrt(norm_diff/(double) (nx*ny));
}

//void linsolve_hc2d_gs_adi(int nx, int ny, double rx, double ry, double **rhs, double **T, double **Tnew)
//{
//
//  // write the Alternating Direction Implicit method here
//  // The argument list may need to change to allow for a work array
//  // It will be similar to the Gauss-Seidel function, except 
//  // that the code-snippet labelled `update the solution' will be
//  // replaced by the following logic: In an outer loop, preform y and x sweeps.
//  // In the sweep along y-lines, for each j, solve a tridiagonal system along x
//  // In the sweep along x-lines, for each i, solve a tridiagonal system along y
//}

//void linsolve_hc2d_gs_rb(int nx, int ny, double rx, double ry, double **rhs, double **T, double **Tnew)
//{
//
//  // write the red-black Gauss-Seidel method here
//  // It will be very similar to the original Gauss-Seidel, except 
//  // that the code-snippet labelled `update the solution' will be
//  // split into two portions
//}

void linsolve_hc2d_gs(int nx, int ny, double rx, double ry, double **rhs, double **T, double **Tnew)
{
  int i, j, k, max_iter;
  double tol, denom, local_diff, norm_diff;

  max_iter = 1000; tol = 1.0e-6;
  denom = 1.0 + 2.0*rx + 2.0*ry;

  for(k=0; k<max_iter;k++)
  {
    // update the solution
    for(i=1; i<nx-1; i++)
     for(j=1; j<ny-1; j++)
       Tnew[i][j] = (rhs[i][j] + rx*Tnew[i-1][j] + rx*T[i+1][j] + ry*Tnew[i][j-1] + ry*T[i][j+1]) /denom;

    // check for convergence
    norm_diff = get_error_norm_2d(nx, ny, T, Tnew);
    if(norm_diff < tol) break;

    // prepare for next iteration
    for(i=0; i<nx; i++)
     for(j=0; j<ny; j++)
       T[i][j] = Tnew[i][j];

  }
  printf("In linsolve_hc2d_gs: %d %e\n", k, norm_diff);
}

void linsolve_hc2d_jacobi(int nx, int ny, double rx, double ry, double **rhs, double **T, double **Tnew)
{
  int i, j, k, max_iter;
  double tol, denom, local_diff, norm_diff;

  max_iter = 1000; tol = 1.0e-6;
  denom = 1.0 + 2.0*rx + 2.0*ry;

  for(k=0; k<max_iter;k++)
  {
    // update the solution
    for(i=1; i<nx-1; i++)
     for(j=1; j<ny-1; j++)
       Tnew[i][j] = (rhs[i][j] + rx*T[i-1][j] + rx*T[i+1][j] + ry*T[i][j-1] + ry*T[i][j+1]) /denom;

    // check for convergence
    norm_diff = get_error_norm_2d(nx, ny, T, Tnew);
    if(norm_diff < tol) break;

    // prepare for next iteration
    for(i=0; i<nx; i++)
     for(j=0; j<ny; j++)
       T[i][j] = Tnew[i][j];

  }
  //printf("In linsolve_hc2d_jacobi: %d %e\n", k, norm_diff);

}

void timestep_BwdEuler(int nx, int ny, double dt, double dx, double dy, double kdiff, double *x, double *y, double **T, double **rhs, double **Tnew)
{

  int i,j;
  double rx, ry;

  // Backward (implicit) Euler scheme
  rx = kdiff*dt/(dx*dx);
  ry = kdiff*dt/(dy*dy);

  // initialize rhs to T at current time level
  for(i=1; i<nx-1; i++)
   for(j=1; j<ny-1; j++)
     rhs[i][j] = T[i][j];

  // boundaries: top and bottom
  for(i=0; i<nx; i++)
  {
    rhs[i][0] = 0.0;
    rhs[i][ny-1] = 0.0;
  }

  // boundaries: left and right
  for(j=0; j<ny; j++)
  {
    rhs[0][j] = 0.0;
    rhs[nx-1][j] = 0.0;
  }

  //// -- comment out all except one of the function calls below
  //linsolve_hc2d_jacobi(nx, ny, rx, ry, rhs, T, Tnew);
  linsolve_hc2d_gs(nx, ny, rx, ry, rhs, T, Tnew);
  //linsolve_hc2d_gs_adi(nx, ny, rx, ry, rhs, T, Tnew);
  //linsolve_hc2d_gs_rb(nx, ny, rx, ry, rhs, T, Tnew);

  // set Dirichlet BCs
  enforce_bcs(nx,ny,x,y,T);

}

void output_soln(int rank, int nx, int ny, int it, double tcurr, double *x, double *y, double **T)
{
  int i,j;
  FILE* fp;
  char fname[100];

  sprintf(fname, "T_x_y_%06d_%04d.dat", it, rank);
  //printf("\n%s\n", fname);

  fp = fopen(fname, "w");
  for(i=0; i<nx; i++)
   for(j=0; j<ny; j++)
      fprintf(fp, "%lf %lf %lf\n", x[i], y[j], T[i][j]);
  fclose(fp);

  printf("Done writing solution for rank = %d, time step = %d, time level = %e\n", rank, it, tcurr);
}

void get_processor_grid_ranks(int rank, int size, int px, int py, int *rank_x, int *rank_y)
{
  *rank_y = rank/px;
  *rank_x = rank - (*rank_y) * px;
}

int main(int argc, char** argv)
{

  int nx, ny, nxglob, nyglob, rank, size, px, py, rank_x, rank_y;
  double *x, *y, **T, **rhs, tst, ten, xstglob, xenglob, ystglob, yenglob, dx, dy, dt, tcurr, kdiff;
  double xst, yst, xen, yen, t_print, xlen, ylen, xlenglob, ylenglob;
  double min_dx_dy, **Tnew, *xleftghost, *xrightghost, *ybotghost, *ytopghost;
  double *sendbuf_x, *sendbuf_y, *recvbuf_x, *recvbuf_y;
  int i, it, num_time_steps, it_print, j, istglob, ienglob, jstglob, jenglob;
  FILE* fid;  
  char debugfname[100];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // read inputs
  if(rank==0)
  {
    fid = fopen("input2d.in", "r");
    fscanf(fid, "%d %d\n", &nxglob, &nyglob);
    fscanf(fid, "%lf %lf %lf %lf\n", &xstglob, &xenglob, &ystglob, &yenglob);
    fscanf(fid, "%lf %lf %lf %lf\n", &tst, &ten, &dt, &t_print);
    fscanf(fid, "%lf\n", &kdiff);
    fscanf(fid, "%d  %d \n", &px, &py);
    fclose(fid);

    // calculate global/local variables
    nx = nxglob/px;                     // nx is local to each rank
    xlen = (xenglob-xstglob)/(double)px;  // xlen is local on each rank
    ny = nyglob/py;                     // ny is local to each rank
    ylen = (yenglob-ystglob)/(double)py;  // ylen is local on each rank

    // prepare for time loop
    num_time_steps = (int)((ten-tst)/dt) + 1; // why add 1 to this?
    it_print = (int) (t_print/dt);            // write out every t_print time units
    num_time_steps = 1;

    printf("Inputs are: %d %d %lf %lf\n", nxglob, nyglob, xstglob, xenglob);
    printf("Inputs are: %lf %lf %lf %lf %lf\n", ystglob, yenglob, tst, ten, kdiff);
    printf("Inputs are: %lf %lf %d %d\n", dt, t_print, px, py);

    if(px*py != size)
    {
      printf("%d %d %d\n", size, px, py);
      printf("\nProcessor grid distribution is not consistent with total number of processors. Stopping now\n");
      exit(0);
    }
  }

  // Use MPI_Bcast to send all the variables read or calculated above 
  // ints    :: nxglob, nyglob, nx, ny num_time_steps, it_print
  int *sendarr_int;
  sendarr_int = malloc(8*sizeof(int));
  if(rank==0)
  {
    sendarr_int[0] = nxglob;         sendarr_int[1] = nx;
    sendarr_int[2] = nyglob;         sendarr_int[3] = ny;
    sendarr_int[4] = num_time_steps; sendarr_int[5] = it_print;
    sendarr_int[6] = px;             sendarr_int[7] = py;
  }
  MPI_Bcast(sendarr_int, 8, MPI_INT, 0, MPI_COMM_WORLD);
  if(rank!=0)
  {
            nxglob = sendarr_int[0];       nx = sendarr_int[1]; 
            nyglob = sendarr_int[2];       ny = sendarr_int[3]; 
    num_time_steps = sendarr_int[4]; it_print = sendarr_int[5];
                px = sendarr_int[6];       py = sendarr_int[7];
  }
  free(sendarr_int);

  get_processor_grid_ranks(rank, size, px, py, &rank_x, &rank_y);

  // doubles :: tst, ten, dt, t_print, xlen, xstglob, xenglob
  // ylen, ystglob, yenglob
  double *sendarr_dbl;
  sendarr_dbl = malloc(10*sizeof(double));
  if(rank==0)
  {
    sendarr_dbl[0] = tst;  sendarr_dbl[1] = ten;     sendarr_dbl[2] = dt;      sendarr_dbl[3] = t_print;
    sendarr_dbl[4] = xlen; sendarr_dbl[5] = xstglob; sendarr_dbl[6] = xenglob;
    sendarr_dbl[7] = ylen; sendarr_dbl[8] = ystglob; sendarr_dbl[9] = yenglob;
  }
  MPI_Bcast(sendarr_dbl, 10, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if(rank!=0)
  {
     tst = sendarr_dbl[0]; ten = sendarr_dbl[1];     dt = sendarr_dbl[2];  t_print = sendarr_dbl[4];
    xlen = sendarr_dbl[4]; xstglob = sendarr_dbl[5]; xenglob = sendarr_dbl[6];
    ylen = sendarr_dbl[7]; ystglob = sendarr_dbl[8]; yenglob = sendarr_dbl[9];
  }
  free(sendarr_dbl);

  // compute index in the global system
  istglob = rank_x * (nxglob/px);
  ienglob = (rank_x+1) * (nxglob/px) - 1;
  jstglob = rank_y * (nyglob/py);
  jenglob = (rank_y+1) * (nyglob/py) - 1;

  // compute st and en limits of the grid
  xst = xstglob + rank_x*xlen;  xen = xst + xlen;
  yst = ystglob + rank_y*ylen;  yen = yst + ylen;

  x = (double *)malloc(nx*sizeof(double));
  y = (double *)malloc(ny*sizeof(double));
  T = (double **)malloc(nx*sizeof(double *));
  for(i=0; i<nx; i++)
    T[i] = (double *)malloc(ny*sizeof(double));
  rhs = (double **)malloc(nx*sizeof(double *));
  for(i=0; i<nx; i++)
    rhs[i] = (double *)malloc(ny*sizeof(double));
  Tnew = (double **)malloc(nx*sizeof(double *));
  for(i=0; i<nx; i++)
    Tnew[i] = (double *)malloc(ny*sizeof(double));

  xleftghost  = (double *)malloc(ny*sizeof(double *));
  xrightghost = (double *)malloc(ny*sizeof(double *));
  ybotghost   = (double *)malloc(nx*sizeof(double *));
  ytopghost   = (double *)malloc(nx*sizeof(double *));

  sendbuf_x  = (double *)malloc(ny*sizeof(double *));
  recvbuf_x  = (double *)malloc(ny*sizeof(double *));
  sendbuf_y  = (double *)malloc(nx*sizeof(double *));
  recvbuf_y  = (double *)malloc(nx*sizeof(double *));

  grid(nx,nxglob,istglob,ienglob,xstglob,xenglob,x,&dx); // initialize the grid in x
  grid(ny,nyglob,jstglob,jenglob,ystglob,yenglob,y,&dy); // initialize the grid in x

  // write debug information -- comment once you are sure the code is working fine
  sprintf(debugfname, "debug_%04d.dat", rank);
  fid = fopen(debugfname, "w");
  fprintf(fid, "\n\n\n--Debug-1- %d %d %d\n", rank, rank_x, rank_y);
  fprintf(fid, "\n--Debug-1- %d %d %d %d\n", nx, nxglob, istglob, ienglob);
  fprintf(fid, "\n--Debug-1- %d %d %d %d\n", ny, nyglob, jstglob, jenglob);
  fprintf(fid, "\n--Debug-2- %lf %lf %lf %lf %lf\n", xst, xen, xstglob, xenglob, xlen);
  fprintf(fid, "\n--Debug-2- %lf %lf %lf %lf %lf\n", yst, yen, ystglob, yenglob, ylen);
  fprintf(fid, "--Writing x grid points--\n");
  for(i=0; i<nx; i++)
    fprintf(fid, "%d %d %d %lf\n", rank, i, i+istglob, x[i]);
  fprintf(fid, "--Done writing x grid points--\n");
  fprintf(fid, "--Writing y grid points--\n");
  for(j=0; j<ny; j++)
    fprintf(fid, "%d %d %d %lf\n", rank, j, j+jstglob, y[j]);
  fprintf(fid, "--Done writing y grid points--\n");
  fclose(fid);  

  set_initial_condition(nx, ny, x, y, T, dx, dy);  // initial condition
  output_soln(rank,nx,ny,0,tst,x,y,T);     // output initial

  // start time stepping loop
  for(it=0; it<num_time_steps; it++)
  {
    tcurr = tst + (double)(it+1) * dt;
    if(rank==0) printf("Working on time step no. %d, time = %lf\n", it, tcurr);

    // Forward (explicit) Euler
    timestep_FwdEuler(rank,size,rank_x,rank_y,px,py,nx,nxglob,ny,nyglob,istglob,ienglob,jstglob,jenglob,dt,dx,dy,xleftghost,xrightghost,ybotghost,ytopghost,kdiff,x,y,T,rhs,sendbuf_x,recvbuf_x,sendbuf_y,recvbuf_y);    // update T

    // Backward (implicit) Euler
    //timestep_BwdEuler(nx,ny,dt,dx,dy,kdiff,x,y,T,rhs,Tnew);    // update T

    // output soln every it_print time steps
    if(it%it_print==0)
      output_soln(rank,nx,ny,it,tcurr,x,y,T);
  }

  //// output soln at the last time step
  //output_soln(nx,ny,it,tcurr,x,y,T);

  ////for(i=0; i<nx; i++)
  ////  free(T[i]);
  ////free(T);
  ////for(i=0; i<nx; i++)
  ////  free(rhs[i]);
  ////free(rhs);
  ////for(i=0; i<nx; i++)
  ////  free(Tnew[i]);
  ////free(Tnew);
  //free(y);
  //free(x);

  MPI_Finalize();
  return 0;
}

