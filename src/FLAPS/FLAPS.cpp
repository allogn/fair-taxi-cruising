#include"lib/stdc++.h"
using namespace std;
#define pb push_back 
#define maxG 625
#define maxCandForP 40
#define inf 65536
using namespace std;
typedef vector<int> VI;
typedef vector<VI> VVI;
typedef pair<double, double> P;
typedef pair<int, int> PI;
bool FindMatch(int i, const VVI &w, VI &mr, VI &mc, VI &seen) {
	for (int j = 0; j < w[i].size(); j++) {
		if (w[i][j] && !seen[j]) {
			seen[j] = true;
			if (mc[j] < 0 || FindMatch(mc[j], w, mr, mc, seen))
			{
				mr[i] = j;
				mc[j] = i;
				return true;
			}
		}
	}
	return false;
}
bool PreFindMatch(int i, const VVI &w, VI &mr, VI &mc, VI &seen) {
	for (int j = 0; j < w[i].size(); j++) {
		if (w[i][j] && !seen[j]) {
			seen[j] = true;
			if (mc[j] < 0 || FindMatch(mc[j], w, mr, mc, seen))
			{
				return true;
			}
		}
	}
	return false;
}
int wn, rn, T, G; double aw, dw; double grida; int gridh; int awIng; double p_b;
vector<P> stat1[maxG]; int cand; double D[maxG];
double Euclidean(double x1, double y1, double x2, double y2)
{
	return (double)sqrt((double)((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2)));
}
typedef struct r {
	int id, g, ing;
	double ox, oy, dis, vr;
	r() {}
	r(int id_, double ox_, double oy_, double dx_, double dy_, double vr_)
	{
		id = id_; dis = Euclidean(ox_, oy_, dx_, dy_); ox = ox_; oy = oy_; vr = vr_;
	}
}req;
typedef struct wor {
	int id, ti, g, ing;
	double lx, ly;
	wor() {}
	wor(int id_, int ti_, double lx_, double ly_)
	{
		id = id_; ti = ti_; lx = lx_; ly = ly_;
	}
}worker;
typedef struct trip {
	int idg, n_new; double Delta, p_new;
	trip(int idg_)
	{
		idg = idg_; n_new = 0; p_new = p_b; Delta = inf;
	}
	trip(int idg_, int n_new_, double p_new_, double Delta_)
	{
		idg = idg_; n_new = n_new_; p_new = p_new_; Delta = Delta_;
	}
}tri;
bool operator<(const tri &a, const tri &b) {
	return a.Delta<b.Delta;
}
typedef struct quad {
	double uti; int id, idg, ing;
	quad(double uti_, int id_, int idg_, int ing_)
	{
		uti = uti_; id = id_; idg = idg_; ing = ing_;
	}
}qua;
bool operator<(const qua &a, const qua &b) {
	return a.uti<b.uti;
}
int pointToGrid(double x, double y)
{
	if (x == 100)x = 99.99;
	if (y == 100)y = 99.99;
	int ix = x / grida;
	int iy = y / grida;
	return iy*gridh + ix;
}
vector<int> neighbor(int ind)
{
	vector<int> nei;
	int st = ind - awIng - gridh * awIng;
	for (int i = 0; i <= 2 * awIng; i++)
	{
		for (int j = 0; j <= 2 * awIng; j++)
		{
			nei.pb(st + i + gridh*j);
		}
	}
	return nei;
}
P Calculate(int g, double p_old, int ng_old, int sizeOfRtg)
{
	double ma = -1, gain_old;
	double p, S, p_;
	int curng = ng_old + 1;
	int i;
	for (i = cand - 1; i >= 0; i--)//restore p from a midterm or bisect
	{
		p = stat1[g][i].first; S = stat1[g][i].second;
		if (p == p_old)
			gain_old = min(sizeOfRtg*p_old*D[g] * S, D[g] * ng_old*p_old);
		double gain = min(sizeOfRtg*p*D[g] * S, D[g] * curng*p);
		if (gain >= ma)
		{
			ma = gain;
			p_ = p;
		}
	}
	return make_pair(p_, max(0.0, ma - gain_old));
}
req gr[maxG][10000]; int grs[maxG];
worker gw[maxG][10000]; int gws[maxG];
vector<PI> idToGridw;
int main()
{
	int u = 0;
	FILE *fp0 = fopen("../data/dat.txt", "r");
	FILE *fp1 = fopen("../data/stat.txt", "r");
	FILE *fp2 = fopen("../data/res.txt", "a");
	// workers, requesters, time periods, grids (cells), radius of each worker, time period per worker
	fscanf(fp0, "%d%d%d%d%lf%lf", &wn, &rn, &T, &G, &aw, &dw);
	gridh = (int)sqrt(G);
	grida = (double)100 / gridh;
	awIng = aw / grida;//aw in grida
	int t_ = 1, t;
	int tot = wn + rn;
	int wtn = 0, rtn = 0;
	//reading statistics
	fscanf(fp1, "%d%lf", &cand, &p_b); // number of candidate prices
	for (int j = 0; j<G; j++)
		for (int i = 0; i<cand; i++)
		{
			double p, S;
			fscanf(fp1, "%lf%lf", &p, &S); // price and acceptance ratio based on historical records
			stat1[j].pb(P(p, S));
		}
	for (int i = 0; i<G; i++)
	{
		fscanf(fp1, "%lf", &D[i]); //coefficients for L_p price estimation
	}
	double res = 0;
	for (int i = 0; i <= tot; i++) // for all workers and customers
	{
		fscanf(fp0, "%d", &t); // time period
		if (t != t_ || i == tot) // if a new period started - run algorithm for previous. otherwise read worker and customer info
		{
			t_ = t;//printf("%d",t);
				   //pricing
			int u0 = clock();
			double p[maxG]; int n[maxG];//final price and n^{tg}, n is number of workers per cell
			for (int j = 0; j<G; j++)
			{
				p[j] = n[j] = 0;
			}
			//construct the bipartite graph
			if (rtn != 0 && wtn != 0)
			{
				VVI w(rtn, vector<int>(wtn, 0));
				for (int j = 0; j<G; j++) // for each cell
				{
					int  gwn = gws[j]; // gws and grs are number of worker/requester in j-th cell
					if (gwn == 0)continue;
					for (int k = 0; k<gwn; k++) // for each worker in the cell
					{
						vector<int> nei;
						nei.clear();
						nei = neighbor(pointToGrid(gw[j][k].lx, gw[j][k].ly));
						for (int l = 0; l<nei.size(); l++) // for each neighbor cell (within worker distance)
						{
							int neigh = nei[l];
							if (neigh<0 || neigh >= G)continue;
							for (int m = 0; m<grs[neigh]; m++) // here additional check
							{
								w[gr[neigh][m].id][gw[j][k].id] = 1; // set 1 to each requester in each neighbor
							}
						}
					}
				}
				//pre-matching
				VI lmate = VI(w.size(), -1);
				VI rmate = VI(w[0].size(), -1);
				//max-heap
				long max_workers = 400;
				long cur_workers = 0;
				priority_queue<tri> H;
				while (H.size())H.pop();
				for (int j = 0; j<G; j++)
				{
					int grn = grs[j];
					if (grn == 0)continue;
					H.push(tri(j));
				}
				while (!H.empty()) // heap with trips (requests)
				{
					tri root = H.top();
					H.pop();
					int curg = root.idg; // current cell
					int sizeOfRtg = grs[curg]; // number of requests per cell
					if (root.Delta != inf)
					{
						n[curg] = root.n_new;
						p[curg] = root.p_new;
						for (int j = 0; j<sizeOfRtg; j++)
						{
							VI seen(w[0].size());
							if (cur_workers < max_workers && FindMatch(gr[curg][j].id, w, lmate, rmate, seen)) {
								cur_workers++;
								break;
							}
						}
					}
					if (root.Delta == 0)
					{
						p[curg] = root.p_new;
						n[curg] = root.n_new;//printf("%d %d %f\n",t_,root.idg,root.p_new);
					}
					else// find no worker =p_b; find a worker =dp; there is no task =0; 
					{
						bool flag = 0;//if(root.Delta==inf)printf("*");
						for (int j = 0; j<sizeOfRtg; j++)//lines22-25
						{
							VI seen(w[0].size());
							if (PreFindMatch(gr[curg][j].id, w, lmate, rmate, seen) && cur_workers < max_workers)
							{
								P tmp = Calculate(curg, root.p_new, root.n_new, sizeOfRtg);
								int tnew = root.n_new + 1;
								H.push(tri(curg, tnew, tmp.first, tmp.second));
								flag = 1;
								cur_workers++;
								break;
							}
						}
						if (flag == 0)//lines20-21
						{
							H.push(tri(curg, root.n_new, root.p_new, 0));
						}
					}

				}
				int u1 = clock();
				u += u1 - u0;
				 //calculate the utility
				priority_queue<qua> match;
				for (int j = 0; j<G; j++)
				{
					int grn = grs[j];
					if (grn == 0)continue;
					for (int k = grn - 1; k >= 0; k--)
					{
						if (gr[j][k].vr>p[j])
							match.push(qua(gr[j][k].dis*p[j], gr[j][k].id, j, k));
						else
						{
							gr[j][k] = gr[j][grs[j] - 1]; grs[j]--;
						}
					}
				}
				VI lm = VI(w.size(), -1);
				VI rm = VI(w[0].size(), -1); priority_queue<PI> rem;
				while (!match.empty())
				{
					qua node = match.top();
					match.pop();
					VI seen(w[0].size());
					if (cur_workers > 0 && FindMatch(node.id, w, lm, rm, seen))
					{
						res += node.uti;
						int ind = node.idg;
						rem.push(PI(node.ing, ind));

						for (int j = 0; j<G; j++)
						{
							int gwn = gws[j];
							if (gwn != 0 && cur_workers > 0)
							{
								for (int k = gwn - 1; k >= 0; k--)
								{
									if (gw[j][k].id == lm[node.id])
									{
										gw[j][k] = gw[j][gws[j] - 1]; gws[j]--;
										cur_workers--;
									}
								}
							}
						}

					}
				}
				while (rem.size())
				{
					PI tmp = rem.top(); rem.pop();
					int ind = tmp.second, ing = tmp.first;
					gr[ind][ing] = gr[ind][grs[ind] - 1]; grs[ind]--;
				}
			}
			for (int j = 0; j<G; j++)
			{
				int gwn = gws[j];
				if (gwn != 0)
				{
					for (int k = gwn - 1; k >= 0; k--)
					{
						if (t > gw[j][k].ti + dw)
						{
							gw[j][k] = gw[j][gws[j] - 1]; gws[j]--;
						}
					}
				}
			}

			//reid
			wtn = rtn = 0;
			for (int j = 0; j<G; j++)
			{
				int grn = grs[j];
				if (grn != 0)
				{
					for (int k = 0; k<grn; k++)
					{
						gr[j][k].id = rtn++;
					}
				}
				int gwn = gws[j];
				if (gwn != 0)
				{
					for (int k = 0; k<gwn; k++)
					{
						idToGridw[wtn].first = j; idToGridw[wtn].second = k;
						gw[j][k].id = wtn++;
					}
				}
			}

			if (i == tot)break;
		}
		char c;
		fscanf(fp0, "%c%c", &c, &c);
		if (c == 'w')
		{
			double x, y;
			fscanf(fp0, "%lf%lf", &x, &y); // timePeriod w x-coordinate y-coordinate
			int ind = pointToGrid(x, y), ing = gws[ind];
			gw[ind][gws[ind]++] = worker(wtn++, t, x, y);
			idToGridw.pb(PI(ind, ing));
		}
		else if (c == 'r')
		{
			double ox, oy, dx, dy, vr;
			//timePeriod r x-coordinateOfOrigin y-coordinateOfOrigin x-coordinateOfDestination y-coordinateOfDestination valuation(vr)
			fscanf(fp0, "%lf%lf%lf%lf%lf", &ox, &oy, &dx, &dy, &vr);
			int ind = pointToGrid(ox, oy), ing = grs[ind];
			gr[ind][grs[ind]++] = req(rtn++, ox, oy, dx, dy, vr);
		}
	}
	
	//	int u1 = clock();
	//	u = u1 - u;
	fprintf(fp2, "MAPS:%f %f\n", res, (double)u / 1000);
	printf("%f %f\n", res, (double)u / 1000);
	return 0;
}
