#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef N
#define N 100
#endif
#ifndef EPS
#define EPS 0.3
#endif
#ifndef BETA
#define BETA 0.7
#endif
#ifndef GAMMA
#define GAMMA 0.5
#endif

static void rhs(const double *z, double *f, double coupling) {
    double mean_x = 0.0;
    for (int i = 0; i < N; ++i) mean_x += z[i];
    mean_x /= N;
    for (int i = 0; i < N; ++i) {
        const double x = z[i], y = z[N + i];
        f[i] = (x * (1.0 - x * x / 3.0) - y) / EPS
               + coupling * (mean_x - x);
        f[N + i] = EPS * (x + BETA - GAMMA * y);
    }
}

static void rk4_step(double *z, double h, double coupling) {
    double k1[2*N], k2[2*N], k3[2*N], k4[2*N], work[2*N];
    rhs(z, k1, coupling);
    for (int i = 0; i < 2*N; ++i) work[i] = z[i] + 0.5*h*k1[i];
    rhs(work, k2, coupling);
    for (int i = 0; i < 2*N; ++i) work[i] = z[i] + 0.5*h*k2[i];
    rhs(work, k3, coupling);
    for (int i = 0; i < 2*N; ++i) work[i] = z[i] + h*k3[i];
    rhs(work, k4, coupling);
    for (int i = 0; i < 2*N; ++i)
        z[i] += h*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i])/6.0;
}

/* Historical representative-based, order-dependent rule. */
static int representative_clusters(const double *z, double eps) {
    int reps[N], count = 0;
    for (int i = 0; i < N; ++i) {
        int found = 0;
        for (int j = 0; j < count; ++j) {
            const int r = reps[j];
            if (fabs(z[i]-z[r]) < eps && fabs(z[N+i]-z[N+r]) < eps) {
                found = 1;
                break;
            }
        }
        if (!found) reps[count++] = i;
    }
    return count;
}

/* Order-independent connected components under the same pairwise threshold. */
static int component_clusters(const double *z, double eps) {
    int parent[N];
    for (int i = 0; i < N; ++i) parent[i] = i;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if (fabs(z[i]-z[j]) < eps && fabs(z[N+i]-z[N+j]) < eps) {
                int a=i, b=j;
                while (parent[a] != a) a=parent[a];
                while (parent[b] != b) b=parent[b];
                if (a != b) parent[b] = a;
            }
        }
    }
    int count = 0;
    for (int i = 0; i < N; ++i) {
        int a=i;
        while (parent[a] != a) a=parent[a];
        if (a == i) ++count;
    }
    return count;
}

static int component_labels(const double *z, double eps, int *labels) {
    int parent[N];
    for (int i = 0; i < N; ++i) parent[i] = i;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if (fabs(z[i]-z[j]) < eps && fabs(z[N+i]-z[N+j]) < eps) {
                int a=i, b=j;
                while (parent[a] != a) a=parent[a];
                while (parent[b] != b) b=parent[b];
                if (a != b) parent[b] = a;
            }
        }
    }
    int roots[N], count=0;
    for (int i = 0; i < N; ++i) {
        int a=i;
        while (parent[a] != a) a=parent[a];
        int label=-1;
        for (int j = 0; j < count; ++j)
            if (roots[j] == a) { label=j; break; }
        if (label < 0) { roots[count]=a; label=count++; }
        labels[i]=label;
    }
    return count;
}

static int partition_pair_mismatches(const int *a, const int *b) {
    int mismatches=0;
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            if ((a[i] == a[j]) != (b[i] == b[j])) ++mismatches;
    return mismatches;
}

static void print_component_sizes(const double *z, double eps) {
    int parent[N], sizes[N] = {0};
    for (int i = 0; i < N; ++i) parent[i] = i;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if (fabs(z[i]-z[j]) < eps && fabs(z[N+i]-z[N+j]) < eps) {
                int a=i, b=j;
                while (parent[a] != a) a=parent[a];
                while (parent[b] != b) b=parent[b];
                if (a != b) parent[b] = a;
            }
        }
    }
    for (int i = 0; i < N; ++i) {
        int a=i;
        while (parent[a] != a) a=parent[a];
        ++sizes[a];
    }
    for (int i = 0; i < N; ++i)
        for (int j = i + 1; j < N; ++j)
            if (sizes[j] > sizes[i]) {
                const int tmp=sizes[i]; sizes[i]=sizes[j]; sizes[j]=tmp;
            }
    printf("component_sizes_descending,1.0e-04");
    for (int i = 0; i < N && sizes[i] > 0; ++i) printf(",%d", sizes[i]);
    putchar('\n');
}

static void print_component_centroids(const double *z, double eps) {
    int labels[N], count = component_labels(z, eps, labels);
    double sx[N] = {0.0}, sy[N] = {0.0};
    int sizes[N] = {0};
    for (int i = 0; i < N; ++i) {
        const int a = labels[i];
        sx[a] += z[i]; sy[a] += z[N+i]; ++sizes[a];
    }
    for (int i = 0; i < count; ++i) {
        sx[i] /= sizes[i]; sy[i] /= sizes[i];
    }
    for (int i = 0; i < count; ++i) for (int j = i+1; j < count; ++j) {
        if (sx[j] < sx[i]) {
            double tx=sx[i], ty=sy[i]; int ts=sizes[i];
            sx[i]=sx[j]; sy[i]=sy[j]; sizes[i]=sizes[j];
            sx[j]=tx; sy[j]=ty; sizes[j]=ts;
        }
    }
    printf("component_centroids,1.0e-04,%d", count);
    for (int i = 0; i < count; ++i)
        printf(",%.17g,%.17g,%d", sx[i], sy[i], sizes[i]);
    putchar('\n');
}

static void print_final_labels(const double *z, double eps) {
    int labels[N];
    const int count = component_labels(z, eps, labels);
    printf("final_labels,1.0e-04,%d", count);
    for (int i = 0; i < N; ++i) printf(",%d", labels[i]);
    putchar('\n');
}

static int read_initial(FILE *input, double *z) {
    char line[4096];
    int count = 0;
    while (fgets(line, sizeof line, input)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        double x, y;
        if (sscanf(line, "%lf %lf", &x, &y) == 2) {
            if (count >= N) return -1;
            z[count] = x;
            z[N + count] = y;
            ++count;
        }
    }
    return count == N ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 10) {
        fprintf(stderr, "usage: %s coupling step_size steps [report_every [final_window [trace_every [snapshot_step [event_every [state_every]]]]]] < initial_values\n", argv[0]);
        return 2;
    }
    const double coupling = strtod(argv[1], NULL);
    const double h = strtod(argv[2], NULL);
    const long steps = strtol(argv[3], NULL, 10);
    const long report_every = argc >= 5 ? strtol(argv[4], NULL, 10) : 0;
    const long final_window = argc >= 6 ? strtol(argv[5], NULL, 10) : 0;
    const long trace_every = argc >= 7 ? strtol(argv[6], NULL, 10) : 0;
    const long snapshot_step = argc >= 8 ? strtol(argv[7], NULL, 10) : 0;
    const long event_every = argc >= 9 ? strtol(argv[8], NULL, 10) : 0;
    const long state_every = argc == 10 ? strtol(argv[9], NULL, 10) : 0;
    if (!(h > 0.0) || steps < 0 || report_every < 0 || final_window < 0 || final_window > steps || trace_every < 0 || snapshot_step < 0 || snapshot_step > steps || event_every < 0 || state_every < 0) {
        fprintf(stderr, "step_size must be positive; counts must be nonnegative\n");
        return 2;
    }
    double z[2*N], snapshot[2*N], previous_event_sample[2*N];
    int have_snapshot=0;
    int have_previous_event_sample=0, quiet_event_samples=0;
    long detected_event_step=0, event_feature_step=0;
    int reference_labels[N], labels[N], have_reference=0;
    int window_labels[N], have_window_reference=0;
    int window_max_pair_mismatches=0;
    double window_max_intra_chebyshev=0.0, window_min_inter_chebyshev=INFINITY;
    double window_mean_x_min=INFINITY, window_mean_x_max=-INFINITY;
    double previous_mean_x=0.0, previous_previous_mean_x=0.0;
    int have_one_mean=0, have_two_means=0, window_mean_x_local_maxima=0;
    long window_samples=0;
    if (read_initial(stdin, z) != 0) {
        fprintf(stderr, "expected exactly 100 non-comment x y rows on stdin\n");
        return 2;
    }
    if (report_every > 0)
        printf("checkpoint_step,h,K,eps,representative_clusters,component_clusters,pair_membership_mismatches\n");
    if (trace_every > 0)
        printf("trace_step,mean_x,sd_x,mean_dx,max_abs_dx,count_abs_dx_gt_1,count_abs_dx_gt_2,count_abs_dx_gt_5,component_clusters_1e_4,fold_x_abs,min_fold_distance,count_fold_near_005,count_fold_near_01\n");
    for (long step = 1; step <= steps; ++step) {
        rk4_step(z, h, coupling);
        if (snapshot_step > 0 && step == snapshot_step) {
            memcpy(snapshot, z, sizeof snapshot);
            have_snapshot=1;
        }
        if (event_every > 0 && step % event_every == 0 && detected_event_step == 0) {
            double derivative[2*N];
            int count5=0, event_clusters;
            rhs(z, derivative, coupling);
            for (int i = 0; i < N; ++i) if (fabs(derivative[i]) > 5.0) ++count5;
            event_clusters=component_clusters(z, 1e-4);
            if (have_previous_event_sample && quiet_event_samples >= 5 && count5 >= 5 && event_clusters >= 90) {
                memcpy(snapshot, previous_event_sample, sizeof snapshot);
                have_snapshot=1;
                detected_event_step=step;
                event_feature_step=step-event_every;
            }
            if (count5 == 0) ++quiet_event_samples; else quiet_event_samples=0;
            memcpy(previous_event_sample, z, sizeof z);
            have_previous_event_sample=1;
        }
        if (trace_every > 0 && step % trace_every == 0) {
            double derivative[2*N], mean_x=0.0, mean_dx=0.0, max_abs_dx=0.0;
            double sum_square=0.0;
            int count1=0, count2=0, count5=0, count_fold_005=0, count_fold_01=0;
            rhs(z, derivative, coupling);
            for (int i = 0; i < N; ++i) {
                mean_x += z[i]; mean_dx += derivative[i];
                const double speed=fabs(derivative[i]);
                if (speed > max_abs_dx) max_abs_dx=speed;
                if (speed > 1.0) ++count1;
                if (speed > 2.0) ++count2;
                if (speed > 5.0) ++count5;
            }
            mean_x /= N; mean_dx /= N;
            const double fold_x_abs = sqrt(fmax(0.0, 1.0 - EPS * coupling));
            double min_fold_distance = INFINITY;
            for (int i = 0; i < N; ++i) {
                const double d = fabs(fabs(z[i]) - fold_x_abs);
                if (d < min_fold_distance) min_fold_distance = d;
                if (d < 0.05) ++count_fold_005;
                if (d < 0.10) ++count_fold_01;
            }
            for (int i = 0; i < N; ++i) {
                const double delta=z[i]-mean_x; sum_square += delta*delta;
            }
            printf("trace,%ld,%.17g,%.17g,%.17g,%.17g,%d,%d,%d,%d,%.17g,%.17g,%d,%d\n", step, mean_x,
                   sqrt(sum_square/(N-1)), mean_dx, max_abs_dx, count1, count2, count5,
                   component_clusters(z, 1e-4), fold_x_abs, min_fold_distance,
                   count_fold_005, count_fold_01);
        }
        if (state_every > 0 && step % state_every == 0) {
            for (int i = 0; i < N; ++i)
                printf("state,%ld,%d,%.17g,%.17g\n", step, i, z[i], z[N+i]);
        }
        if (report_every > 0 && step % report_every == 0) {
            const int count=component_labels(z, 1e-4, labels);
            if (!have_reference) {
                memcpy(reference_labels, labels, sizeof labels);
                have_reference=1;
            }
            printf("%ld,%.17g,%.17g,1.0e-04,%d,%d,%d\n", step, h, coupling,
                   representative_clusters(z, 1e-4), count,
                   partition_pair_mismatches(reference_labels, labels));
        }
        if (final_window > 0 && step >= steps-final_window+1) {
            if (!have_window_reference) {
                component_labels(z, 1e-4, window_labels);
                have_window_reference=1;
            }
            component_labels(z, 1e-4, labels);
            const int mismatches=partition_pair_mismatches(window_labels, labels);
            if (mismatches > window_max_pair_mismatches) window_max_pair_mismatches=mismatches;
            for (int i = 0; i < N; ++i) {
                for (int j = i+1; j < N; ++j) {
                    const double dx=fabs(z[i]-z[j]), dy=fabs(z[N+i]-z[N+j]);
                    const double d=dx > dy ? dx : dy;
                    if (window_labels[i] == window_labels[j]) {
                        if (d > window_max_intra_chebyshev) window_max_intra_chebyshev=d;
                    } else if (d < window_min_inter_chebyshev) {
                        window_min_inter_chebyshev=d;
                    }
                }
            }
            double mean_x=0.0;
            for (int i = 0; i < N; ++i) mean_x += z[i];
            mean_x /= N;
            if (mean_x < window_mean_x_min) window_mean_x_min=mean_x;
            if (mean_x > window_mean_x_max) window_mean_x_max=mean_x;
            if (have_two_means && previous_mean_x > previous_previous_mean_x
                               && previous_mean_x >= mean_x)
                ++window_mean_x_local_maxima;
            previous_previous_mean_x=previous_mean_x;
            previous_mean_x=mean_x;
            if (have_one_mean) have_two_means=1;
            have_one_mean=1;
            ++window_samples;
        }
    }

    const double thresholds[] = {1e-3, 1e-4, 1e-5, 1e-6, 1e-8};
    printf("final_step,h,K,eps,representative_clusters,component_clusters\n");
    for (size_t i = 0; i < sizeof thresholds / sizeof thresholds[0]; ++i) {
        const double eps = thresholds[i];
        printf("%ld,%.17g,%.17g,%.1e,%d,%d\n", steps, h, coupling, eps,
               representative_clusters(z, eps), component_clusters(z, eps));
    }
    print_component_sizes(z, 1e-4);
    print_component_centroids(z, 1e-4);
    print_final_labels(z, 1e-4);
    if (have_snapshot) {
        int final_labels[N];
        double snapshot_derivative[2*N];
        component_labels(z, 1e-4, final_labels);
        rhs(snapshot, snapshot_derivative, coupling);
        printf("event_summary,%ld,%ld\n", detected_event_step, event_feature_step);
        printf("snapshot_header,index,x,y,dx,final_cluster_label\n");
        for (int i = 0; i < N; ++i)
            printf("snapshot,%d,%.17g,%.17g,%.17g,%d\n", i, snapshot[i],
                   snapshot[N+i], snapshot_derivative[i], final_labels[i]);
    }
    if (final_window > 0)
        printf("window_summary,%ld,%ld,%d,%.17g,%.17g,%.17g,%.17g,%d\n", final_window, window_samples,
               window_max_pair_mismatches, window_max_intra_chebyshev,
               window_min_inter_chebyshev, window_mean_x_min, window_mean_x_max,
               window_mean_x_local_maxima);
    return 0;
}
