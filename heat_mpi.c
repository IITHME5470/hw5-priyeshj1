#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

typedef struct {
    int rank;
    int nprocs;
    MPI_Comm comm_cart;
    int px, py;
    int coords[2];
    int nbrleft, nbrright, nbrtop, nbrbot;
    int nx_global, ny_global;
    int nx_local, ny_local;
    int xst_global_idx, yst_global_idx;
    double *x_local, *y_local;
    double **T_local;
    double **rhs_local;
} ParallelInfo;

double** allocate_2d_array(int rows_interior, int cols_interior) {
    int i;
    int total_rows = rows_interior + 2;
    int total_cols = cols_interior + 2;
    double **arr = (double **)malloc(total_rows * sizeof(double *));
    if (arr == NULL) return NULL;
    double *data = (double *)calloc(total_rows * total_cols, sizeof(double));
    if (data == NULL) { free(arr); return NULL; }
    for (i = 0; i < total_rows; i++) { arr[i] = &(data[i * total_cols]); }
    return arr;
}

void free_2d_array(double **arr) {
    if (arr != NULL) {
        if (arr[0] != NULL) { free(arr[0]); }
        free(arr);
    }
}

void local_grid(ParallelInfo *p_info, double xst_global, double xen_global, double yst_global, double yen_global, double *dx, double *dy) {
    int i, j;
    *dx = (p_info->nx_global > 1) ? (xen_global - xst_global) / (double)(p_info->nx_global - 1) : 0;
    *dy = (p_info->ny_global > 1) ? (yen_global - yst_global) / (double)(p_info->ny_global - 1) : 0;
    for (i = 0; i < p_info->nx_local; i++) {
        p_info->x_local[i] = xst_global + (double)(p_info->xst_global_idx + i) * (*dx);
    }
    for (j = 0; j < p_info->ny_local; j++) {
        p_info->y_local[j] = yst_global + (double)(p_info->yst_global_idx + j) * (*dy);
    }
}

void enforce_bcs_local(ParallelInfo *p_info) {
    int i, j;
    int nx_loc = p_info->nx_local;
    int ny_loc = p_info->ny_local;
    double **T = p_info->T_local;
    if (p_info->nbrleft == MPI_PROC_NULL) {
        for (j = 0; j <= ny_loc + 1; j++) { T[0][j] = 0.0; }
    }
    if (p_info->nbrright == MPI_PROC_NULL) {
         for (j = 0; j <= ny_loc + 1; j++) { T[nx_loc + 1][j] = 0.0; }
    }
    if (p_info->nbrbot == MPI_PROC_NULL) {
        for (i = 0; i <= nx_loc + 1; i++) { T[i][0] = 0.0; }
    }
    if (p_info->nbrtop == MPI_PROC_NULL) {
         for (i = 0; i <= nx_loc + 1; i++) { T[i][ny_loc + 1] = 0.0; }
    }
}

void set_initial_condition_local(ParallelInfo *p_info, double dx, double dy) {
    int i, j, li, lj;
    double del = 1.0;
    double x_glob, y_glob;
    if (dx <= 0.0 || dy <= 0.0) {
         enforce_bcs_local(p_info); return;
    }
    for (i = 0; i < p_info->nx_local; i++) {
        li = i + 1;
        x_glob = p_info->x_local[i];
        for (j = 0; j < p_info->ny_local; j++) {
            lj = j + 1;
            y_glob = p_info->y_local[j];
            p_info->T_local[li][lj] = 0.25 * (tanh((x_glob - 0.4) / (del * dx)) - tanh((x_glob - 0.6) / (del * dx)))
                                      * (tanh((y_glob - 0.4) / (del * dy)) - tanh((y_glob - 0.6) / (del * dy)));
        }
    }
    enforce_bcs_local(p_info);
}

void halo_exchange(ParallelInfo *p_info) {
    int nx_loc = p_info->nx_local;
    int ny_loc = p_info->ny_local;
    double **T = p_info->T_local;
    MPI_Comm comm = p_info->comm_cart;
    MPI_Status status;
    MPI_Datatype col_exch_type;

    MPI_Sendrecv(&T[nx_loc][1], ny_loc, MPI_DOUBLE, p_info->nbrtop, 201,
                 &T[0][1],      ny_loc, MPI_DOUBLE, p_info->nbrbot, 201, comm, &status);
    MPI_Sendrecv(&T[1][1],        ny_loc, MPI_DOUBLE, p_info->nbrbot, 202,
                 &T[nx_loc+1][1], ny_loc, MPI_DOUBLE, p_info->nbrtop, 202, comm, &status);

    MPI_Type_vector(nx_loc, 1, ny_loc + 2, MPI_DOUBLE, &col_exch_type);
    MPI_Type_commit(&col_exch_type);
    MPI_Sendrecv(&T[1][ny_loc], 1, col_exch_type, p_info->nbrright, 203,
                 &T[1][0],     1, col_exch_type, p_info->nbrleft,  203, comm, &status);
    MPI_Sendrecv(&T[1][1],       1, col_exch_type, p_info->nbrleft,  204,
                 &T[1][ny_loc+1], 1, col_exch_type, p_info->nbrright, 204, comm, &status);
    MPI_Type_free(&col_exch_type);
}

void timestep_FwdEuler_local(ParallelInfo *p_info, double dt, double dx, double dy, double kdiff) {
    int i, j;
    int nx_loc = p_info->nx_local;
    int ny_loc = p_info->ny_local;
    double **T = p_info->T_local;
    double **rhs = p_info->rhs_local;
    double dxsq = (dx > 0) ? dx * dx : 1.0;
    double dysq = (dy > 0) ? dy * dy : 1.0;

    halo_exchange(p_info);

    if (dxsq > 0 && dysq > 0) {
        for (i = 1; i <= nx_loc; i++) {
            for (j = 1; j <= ny_loc; j++) {
                rhs[i][j] = kdiff * (T[i + 1][j] + T[i - 1][j] - 2.0 * T[i][j]) / dxsq +
                            kdiff * (T[i][j + 1] + T[i][j - 1] - 2.0 * T[i][j]) / dysq;
            }
        }
        for (i = 1; i <= nx_loc; i++) {
            for (j = 1; j <= ny_loc; j++) {
                T[i][j] = T[i][j] + dt * rhs[i][j];
            }
        }
    }
    enforce_bcs_local(p_info);
}

void output_soln_local(ParallelInfo *p_info, int it, double tcurr, double dx, double dy, double xst_global, double yst_global) {
    int i, j, li, lj;
    FILE* fp;
    char fname[100];
    double x_glob, y_glob;
    sprintf(fname, "T_x_y_%06d_rank_%d.dat", it, p_info->rank);
    fp = fopen(fname, "w");
    if (fp == NULL) {
        fprintf(stderr, "Rank %d Error: Could not open file %s for writing.\n", p_info->rank, fname);
        MPI_Abort(p_info->comm_cart, 1); return;
    }
    for (i = 0; i < p_info->nx_local; i++) {
        li = i + 1;
        x_glob = xst_global + (double)(p_info->xst_global_idx + i) * dx;
        for (j = 0; j < p_info->ny_local; j++) {
            lj = j + 1;
            y_glob = yst_global + (double)(p_info->yst_global_idx + j) * dy;
            fprintf(fp, "%lf %lf %lf\n", x_glob, y_glob, p_info->T_local[li][lj]);
        }
    }
    fclose(fp);
}

int main(int argc, char *argv[]) {
    ParallelInfo p_info;
    int nx_global, ny_global;
    double tst, ten, xst, xen, yst, yen, dx, dy, dt, tcurr, kdiff;
    double min_dx_dy_global;
    int i, it, num_time_steps, it_print;
    FILE* fp_in;
    double time_start = 0.0, time_end = 0.0, elapsed_total = 0.0, max_elapsed_total = 0.0;
    int px_in = 0, py_in = 0;
    double step_start_time, step_end_time, step_duration, max_step_duration;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &p_info.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p_info.nprocs);

    if (p_info.rank == 0) {
        fp_in = fopen("input2d.in", "r");
        if (fp_in == NULL) { fprintf(stderr, "Rank 0 Error: Input file missing.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fscanf(fp_in, "%d %d\n", &nx_global, &ny_global) != 2) 
        { fprintf(stderr,"Input Error (nx ny)\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fscanf(fp_in, "%lf %lf %lf %lf\n", &xst, &xen, &yst, &yen) != 4)
        { fprintf(stderr,"Input Error (xst xen yst yen)\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fscanf(fp_in, "%lf %lf\n", &tst, &ten) != 2)
        { fprintf(stderr,"Input Error (tst ten)\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fscanf(fp_in, "%lf\n", &kdiff) != 1) 
        { fprintf(stderr,"Input Error (kdiff)\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        if (fscanf(fp_in, "%d %d", &px_in, &py_in) != 2) 
        { fprintf(stderr,"Input Error (px py)\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        fclose(fp_in);
        if (px_in * py_in != p_info.nprocs) {
             fprintf(stderr, "Error: File Px*Py (%d*%d=%d) != Nprocs (%d).\n", px_in, py_in, px_in*py_in, p_info.nprocs);
             MPI_Abort(MPI_COMM_WORLD, 1);
        }
         if (px_in <= 0 || py_in <=0) { fprintf(stderr, "Error: Px, Py must be positive.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        printf("Inputs are: %d %.6f %.6f %.6f %.6f %.6f\n", nx_global, xst, xen, tst, ten, kdiff);
        printf("Inputs are: %d %.6f %.6f\n", ny_global, yst, yen);
    }

    MPI_Bcast(&nx_global, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ny_global, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&xst, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&xen, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&yst, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&yen, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tst, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ten, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&kdiff, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&px_in, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&py_in, 1, MPI_INT, 0, MPI_COMM_WORLD);
    p_info.nx_global = nx_global; p_info.ny_global = ny_global;

    int dims[2] = {px_in, py_in}; int periods[2] = {0, 0}; int reorder = 1;
    p_info.px = dims[0]; p_info.py = dims[1];
    if (nx_global % p_info.px != 0 || ny_global % p_info.py != 0) {
        if (p_info.rank == 0) fprintf(stderr, "Error: Grid not divisible by process layout.\n");
        MPI_Finalize(); return 1;
    }
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &p_info.comm_cart);
    MPI_Cart_coords(p_info.comm_cart, p_info.rank, 2, p_info.coords);
    MPI_Cart_shift(p_info.comm_cart, 1, 1, &p_info.nbrleft, &p_info.nbrright);
    MPI_Cart_shift(p_info.comm_cart, 0, 1, &p_info.nbrbot, &p_info.nbrtop);

    p_info.nx_local = nx_global / p_info.px;
    p_info.ny_local = ny_global / p_info.py;
    p_info.xst_global_idx = p_info.coords[0] * p_info.nx_local;
    p_info.yst_global_idx = p_info.coords[1] * p_info.ny_local;

    p_info.x_local = (double *)malloc(p_info.nx_local * sizeof(double));
    p_info.y_local = (double *)malloc(p_info.ny_local * sizeof(double));
    p_info.T_local = allocate_2d_array(p_info.nx_local, p_info.ny_local);
    p_info.rhs_local = allocate_2d_array(p_info.nx_local, p_info.ny_local);
    if (!p_info.x_local || !p_info.y_local || !p_info.T_local || !p_info.rhs_local) {
        fprintf(stderr, "Rank %d Error: Memory allocation failed.\n", p_info.rank);
        MPI_Abort(p_info.comm_cart, 1);
    }

    local_grid(&p_info, xst, xen, yst, yen, &dx, &dy);
    set_initial_condition_local(&p_info, dx, dy);

    min_dx_dy_global = fmin(dx, dy);
    dt = (kdiff > 0 && min_dx_dy_global > 0) ? (0.1 / kdiff * (min_dx_dy_global * min_dx_dy_global)) : 1e-6;
    num_time_steps = (ten > tst && dt > 0) ? (int)((ten - tst) / dt) + 1 : 0;
    it_print = (num_time_steps > 10) ? (num_time_steps / 10) : 1;
    if (it_print <= 0) it_print = 1;

    tcurr = tst;
     if (p_info.rank == 0) {
        printf("Done writing solution for time step = %d, time level = %e\n", 0, tcurr);
    }
    output_soln_local(&p_info, 0, tcurr, dx, dy, xst, yst);

    MPI_Barrier(p_info.comm_cart);
    time_start = MPI_Wtime();

    for (it = 1; it <= num_time_steps; it++) {
        tcurr = tst + (double)it * dt;

        step_start_time = MPI_Wtime();
        timestep_FwdEuler_local(&p_info, dt, dx, dy, kdiff);
        step_end_time = MPI_Wtime();
        step_duration = step_end_time - step_start_time;

        if (it % it_print == 0) {
            MPI_Reduce(&step_duration, &max_step_duration, 1, MPI_DOUBLE, MPI_MAX, 0, p_info.comm_cart);
            if (p_info.rank == 0) {
                 printf("Done writing solution for time step = %d, time level = %e (step took %.6e s)\n", it, tcurr, max_step_duration);
            }
            output_soln_local(&p_info, it, tcurr, dx, dy, xst, yst);
        }
    }

    MPI_Barrier(p_info.comm_cart);
    time_end = MPI_Wtime();
    elapsed_total = time_end - time_start;

    if ((num_time_steps % it_print) != 0 && num_time_steps > 0) {
       tcurr = tst + (double)num_time_steps * dt;
       MPI_Reduce(&step_duration, &max_step_duration, 1, MPI_DOUBLE, MPI_MAX, 0, p_info.comm_cart);
        if (p_info.rank == 0) {
           printf("Done writing solution for time step = %d, time level = %e (step took %.6e s)\n", num_time_steps, tcurr, max_step_duration);
       }
       output_soln_local(&p_info, num_time_steps, tcurr, dx, dy, xst, yst);
    }

    free(p_info.x_local);
    free(p_info.y_local);
    free_2d_array(p_info.T_local);
    free_2d_array(p_info.rhs_local);
    MPI_Comm_free(&p_info.comm_cart);
    MPI_Finalize();
    return 0;
}