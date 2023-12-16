// Voro++, a 3D cell-based Voronoi library
//
// Author   : Chris H. Rycroft (Harvard University / LBL)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

/** \file cell.cc
 * \brief Function implementations for the voronoicell and related classes. */

#include <cmath>
#include <cstring>

#include "config.hh"
#include "common.hh"
#include "cell.hh"

namespace voro {

/** Constructs a Voronoi cell and sets up the initial memory. */
voronoicell_base::voronoicell_base(double max_len_sq) :
	current_vertices(init_vertices), current_vertex_order(init_vertex_order),
	current_delete_size(init_delete_size), current_delete2_size(init_delete2_size),
	current_xsearch_size(init_xsearch_size),
	ed(new int*[current_vertices]), nu(new int[current_vertices]),
	mask(new unsigned int[current_vertices]),
	pts(new double[current_vertices<<2]), tol(tolerance*max_len_sq),
	tol_cu(tol*sqrt(tol)), big_tol(big_tolerance_fac*tol), mem(new int[current_vertex_order]),
	mec(new int[current_vertex_order]),
	mep(new int*[current_vertex_order]), ds(new int[current_delete_size]),
	stacke(ds+current_delete_size), ds2(new int[current_delete2_size]),
	stacke2(ds2+current_delete2_size), xse(new int[current_xsearch_size]),
	stacke3(xse+current_xsearch_size), maskc(0) {
	int i;
	for(i=0;i<current_vertices;i++) mask[i]=0;
	for(i=0;i<3;i++) {
		mem[i]=init_n_vertices;mec[i]=0;
		mep[i]=new int[init_n_vertices*((i<<1)+1)];
	}
	mem[3]=init_3_vertices;mec[3]=0;
	mep[3]=new int[init_3_vertices*7];
	for(i=4;i<current_vertex_order;i++) {
		mem[i]=init_n_vertices;mec[i]=0;
		mep[i]=new int[init_n_vertices*((i<<1)+1)];
	}
}

/** The voronoicell destructor deallocates all the dynamic memory. */
voronoicell_base::~voronoicell_base() {
	for(int i=current_vertex_order-1;i>=0;i--) if(mem[i]>0) delete [] mep[i];
	delete [] xse;
	delete [] ds2;delete [] ds;
	delete [] mep;delete [] mec;
	delete [] mem;delete [] pts;
	delete [] mask;
	delete [] nu;delete [] ed;
}

/** Ensures that enough memory is allocated prior to carrying out a copy.
 * \param[in] vc a reference to the specialized version of the calling class.
 * \param[in] vb a pointered to the class to be copied. */
template<class vc_class>
void voronoicell_base::check_memory_for_copy(vc_class &vc,voronoicell_base* vb) {
	while(current_vertex_order<vb->current_vertex_order) add_memory_vorder(vc);
	for(int i=0;i<current_vertex_order;i++) while(mem[i]<vb->mec[i]) add_memory(vc,i);
	while(current_vertices<vb->p) add_memory_vertices(vc);
}

/** Copies the vertex and edge information from another class. The routine
 * assumes that enough memory is available for the copy.
 * \param[in] vb a pointer to the class to copy. */
void voronoicell_base::copy(voronoicell_base* vb) {
	int i,j;
	p=vb->p;up=0;
	for(i=0;i<current_vertex_order;i++) {
		mec[i]=vb->mec[i];
		for(j=0;j<mec[i]*(2*i+1);j++) mep[i][j]=vb->mep[i][j];
		for(j=0;j<mec[i]*(2*i+1);j+=2*i+1) ed[mep[i][j+2*i]]=mep[i]+j;
	}
	for(i=0;i<p;i++) nu[i]=vb->nu[i];
	for(i=0;i<(p<<2);i++) pts[i]=vb->pts[i];
}

/** Copies the information from another voronoicell class into this
 * class, extending memory allocation if necessary.
 * \param[in] c the class to copy. */
void voronoicell_neighbor::operator=(voronoicell &c) {
	voronoicell_base *vb=((voronoicell_base*) &c);
	check_memory_for_copy(*this,vb);copy(vb);
	int i,j;
	for(i=0;i<c.current_vertex_order;i++) {
		for(j=0;j<c.mec[i]*i;j++) mne[i][j]=0;
		for(j=0;j<c.mec[i];j++) ne[c.mep[i][(2*i+1)*j+2*i]]=mne[i]+(j*i);
	}
}

/** Copies the information from another voronoicell_neighbor class into this
 * class, extending memory allocation if necessary.
 * \param[in] c the class to copy. */
void voronoicell_neighbor::operator=(voronoicell_neighbor &c) {
	voronoicell_base *vb=((voronoicell_base*) &c);
	check_memory_for_copy(*this,vb);copy(vb);
	int i,j;
	for(i=0;i<c.current_vertex_order;i++) {
		for(j=0;j<c.mec[i]*i;j++) mne[i][j]=c.mne[i][j];
		for(j=0;j<c.mec[i];j++) ne[c.mep[i][(2*i+1)*j+2*i]]=mne[i]+(j*i);
	}
}

/** Translates the vertices of the Voronoi cell by a given vector.
 * \param[in] (x,y,z) the coordinates of the vector. */
void voronoicell_base::translate(double x,double y,double z) {
	x*=2;y*=2;z*=2;
	double *ptsp=pts;
	while(ptsp<pts+(p<<2)) {
		*(ptsp++)+=x;*(ptsp++)+=y;*ptsp+=z;ptsp+=2;
	}
}

/** Increases the memory storage for a particular vertex order, by increasing
 * the size of the of the corresponding mep array. If the arrays already exist,
 * their size is doubled; if they don't exist, then new ones of size
 * init_n_vertices are allocated. The routine also ensures that the pointers in
 * the ed array are updated, by making use of the back pointers. For the cases
 * where the back pointer has been temporarily overwritten in the marginal
 * vertex code, the auxiliary delete stack is scanned to find out how to update
 * the ed value. If the template has been instantiated with the neighbor
 * tracking turned on, then the routine also reallocates the corresponding mne
 * array.
 * \param[in] i the order of the vertex memory to be increased. */
template<class vc_class>
void voronoicell_base::add_memory(vc_class &vc,int i) {
	int s=(i<<1)+1;
	if(mem[i]==0) {
		vc.n_allocate(i,init_n_vertices);
		mep[i]=new int[init_n_vertices*s];
		mem[i]=init_n_vertices;
#if VOROPP_VERBOSE >=2
		fprintf(stderr,"Order %d vertex memory created\n",i);
#endif
	} else {
		int j=0,k,*l;
		mem[i]<<=1;
		if(mem[i]>max_n_vertices) voro_fatal_error("Point memory allocation exceeded absolute maximum",VOROPP_MEMORY_ERROR);
#if VOROPP_VERBOSE >=2
		fprintf(stderr,"Order %d vertex memory scaled up to %d\n",i,mem[i]);
#endif
		l=new int[s*mem[i]];
		int m=0;
		vc.n_allocate_aux1(i);
		while(j<s*mec[i]) {
			k=mep[i][j+(i<<1)];
			if(k>=0) {
				ed[k]=l+j;
				vc.n_set_to_aux1_offset(k,m);
			} else {
				int *dsp;
				for(dsp=ds2;dsp<stackp2;dsp++) {
					if(ed[*dsp]==mep[i]+j) {
						ed[*dsp]=l+j;
						vc.n_set_to_aux1_offset(*dsp,m);
						break;
					}
				}
				if(dsp==stackp2) {
					for(dsp=xse;dsp<stackp3;dsp++) {
						if(ed[*dsp]==mep[i]+j) {
							ed[*dsp]=l+j;
							vc.n_set_to_aux1_offset(*dsp,m);
							break;
						}
					}
					if(dsp==stackp3) voro_fatal_error("Couldn't relocate dangling pointer",VOROPP_INTERNAL_ERROR);
				}
#if VOROPP_VERBOSE >=3
				fputs("Relocated dangling pointer",stderr);
#endif
			}
			for(k=0;k<s;k++,j++) l[j]=mep[i][j];
			for(k=0;k<i;k++,m++) vc.n_copy_to_aux1(i,m);
		}
		delete [] mep[i];
		mep[i]=l;
		vc.n_switch_to_aux1(i);
	}
}

/** Doubles the maximum number of vertices allowed, by reallocating the ed, nu,
 * and pts arrays. If the allocation exceeds the absolute maximum set in
 * max_vertices, then the routine exits with a fatal error. If the template has
 * been instantiated with the neighbor tracking turned on, then the routine
 * also reallocates the ne array. */
template<class vc_class>
void voronoicell_base::add_memory_vertices(vc_class &vc) {
	int i=(current_vertices<<1),j,**pp,*pnu;
	unsigned int* pmask;
	if(i>max_vertices) voro_fatal_error("Vertex memory allocation exceeded absolute maximum",VOROPP_MEMORY_ERROR);
#if VOROPP_VERBOSE >=2
	fprintf(stderr,"Vertex memory scaled up to %d\n",i);
#endif
	double *ppts;
	pp=new int*[i];
	for(j=0;j<current_vertices;j++) pp[j]=ed[j];
	delete [] ed;ed=pp;
	vc.n_add_memory_vertices(i);
	pnu=new int[i];
	for(j=0;j<current_vertices;j++) pnu[j]=nu[j];
	delete [] nu;nu=pnu;
	pmask=new unsigned int[i];
	for(j=0;j<current_vertices;j++) pmask[j]=mask[j];
	while(j<i) pmask[j++]=0;
	delete [] mask;mask=pmask;
	ppts=new double[i<<2];
	for(j=0;j<(current_vertices<<2);j++) ppts[j]=pts[j];
	delete [] pts;pts=ppts;
	current_vertices=i;
}

/** Doubles the maximum allowed vertex order, by reallocating mem, mep, and mec
 * arrays. If the allocation exceeds the absolute maximum set in
 * max_vertex_order, then the routine causes a fatal error. If the template has
 * been instantiated with the neighbor tracking turned on, then the routine
 * also reallocates the mne array. */
template<class vc_class>
void voronoicell_base::add_memory_vorder(vc_class &vc) {
	int i=(current_vertex_order<<1),j,*p1,**p2;
	if(i>max_vertex_order) voro_fatal_error("Vertex order memory allocation exceeded absolute maximum",VOROPP_MEMORY_ERROR);
#if VOROPP_VERBOSE >=2
	fprintf(stderr,"Vertex order memory scaled up to %d\n",i);
#endif
	p1=new int[i];
	for(j=0;j<current_vertex_order;j++) p1[j]=mem[j];
	while(j<i) p1[j++]=0;
	delete [] mem;mem=p1;
	p2=new int*[i];
	for(j=0;j<current_vertex_order;j++) p2[j]=mep[j];
	delete [] mep;mep=p2;
	p1=new int[i];
	for(j=0;j<current_vertex_order;j++) p1[j]=mec[j];
	while(j<i) p1[j++]=0;
	delete [] mec;mec=p1;
	vc.n_add_memory_vorder(i);
	current_vertex_order=i;
}

/** Doubles the size allocation of the main delete stack. If the allocation
 * exceeds the absolute maximum set in max_delete_size, then routine causes a
 * fatal error. */
void voronoicell_base::add_memory_ds() {
	current_delete_size<<=1;
	if(current_delete_size>max_delete_size) voro_fatal_error("Delete stack 1 memory allocation exceeded absolute maximum",VOROPP_MEMORY_ERROR);
#if VOROPP_VERBOSE >=2
	fprintf(stderr,"Delete stack 1 memory scaled up to %d\n",current_delete_size);
#endif
	int *dsn=new int[current_delete_size],*dsnp=dsn,*dsp=ds;
	while(dsp<stackp) *(dsnp++)=*(dsp++);
	delete [] ds;ds=dsn;stackp=dsnp;
	stacke=ds+current_delete_size;
}

/** Doubles the size allocation of the auxiliary delete stack. If the
 * allocation exceeds the absolute maximum set in max_delete2_size, then the
 * routine causes a fatal error. */
void voronoicell_base::add_memory_ds2() {
	current_delete2_size<<=1;
	if(current_delete2_size>max_delete2_size) voro_fatal_error("Delete stack 2 memory allocation exceeded absolute maximum",VOROPP_MEMORY_ERROR);
#if VOROPP_VERBOSE >=2
	fprintf(stderr,"Delete stack 2 memory scaled up to %d\n",current_delete2_size);
#endif
	int *dsn=new int[current_delete2_size],*dsnp=dsn,*dsp=ds2;
	while(dsp<stackp2) *(dsnp++)=*(dsp++);
	delete [] ds2;ds2=dsn;stackp2=dsnp;
	stacke2=ds2+current_delete2_size;
}

/** Doubles the size allocation of the auxiliary delete stack. If the
 * allocation exceeds the absolute maximum set in max_delete2_size, then the
 * routine causes a fatal error. */
void voronoicell_base::add_memory_xse() {
	current_xsearch_size<<=1;
	if(current_xsearch_size>max_xsearch_size) voro_fatal_error("Extra search stack memory allocation exceeded absolute maximum",VOROPP_MEMORY_ERROR);
#if VOROPP_VERBOSE >=2
	fprintf(stderr,"Extra search stack memory scaled up to %d\n",current_xsearch_size);
#endif
	int *dsn=new int[current_xsearch_size],*dsnp=dsn,*dsp=xse;
	while(dsp<stackp3) *(dsnp++)=*(dsp++);
	delete [] xse;xse=dsn;stackp3=dsnp;
	stacke3=xse+current_xsearch_size;
}

/** Initializes a Voronoi cell as a rectangular box with the given dimensions.
 * \param[in] (xmin,xmax) the minimum and maximum x coordinates.
 * \param[in] (ymin,ymax) the minimum and maximum y coordinates.
 * \param[in] (zmin,zmax) the minimum and maximum z coordinates. */
void voronoicell_base::init_base(double xmin,double xmax,double ymin,double ymax,double zmin,double zmax) {
	for(int i=0;i<current_vertex_order;i++) mec[i]=0;
	up=0;
	mec[3]=p=8;xmin*=2;xmax*=2;ymin*=2;ymax*=2;zmin*=2;zmax*=2;
	*pts=xmin;pts[1]=ymin;pts[2]=zmin;
	pts[4]=xmax;pts[5]=ymin;pts[6]=zmin;
	pts[8]=xmin;pts[9]=ymax;pts[10]=zmin;
	pts[12]=xmax;pts[13]=ymax;pts[14]=zmin;
	pts[16]=xmin;pts[17]=ymin;pts[18]=zmax;
	pts[20]=xmax;pts[21]=ymin;pts[22]=zmax;
	pts[24]=xmin;pts[25]=ymax;pts[26]=zmax;
	pts[28]=xmax;pts[29]=ymax;pts[30]=zmax;
	int *q=mep[3];
	*q=1;q[1]=4;q[2]=2;q[3]=2;q[4]=1;q[5]=0;q[6]=0;
	q[7]=3;q[8]=5;q[9]=0;q[10]=2;q[11]=1;q[12]=0;q[13]=1;
	q[14]=0;q[15]=6;q[16]=3;q[17]=2;q[18]=1;q[19]=0;q[20]=2;
	q[21]=2;q[22]=7;q[23]=1;q[24]=2;q[25]=1;q[26]=0;q[27]=3;
	q[28]=6;q[29]=0;q[30]=5;q[31]=2;q[32]=1;q[33]=0;q[34]=4;
	q[35]=4;q[36]=1;q[37]=7;q[38]=2;q[39]=1;q[40]=0;q[41]=5;
	q[42]=7;q[43]=2;q[44]=4;q[45]=2;q[46]=1;q[47]=0;q[48]=6;
	q[49]=5;q[50]=3;q[51]=6;q[52]=2;q[53]=1;q[54]=0;q[55]=7;
	*ed=q;ed[1]=q+7;ed[2]=q+14;ed[3]=q+21;
	ed[4]=q+28;ed[5]=q+35;ed[6]=q+42;ed[7]=q+49;
	*nu=nu[1]=nu[2]=nu[3]=nu[4]=nu[5]=nu[6]=nu[7]=3;
}

/** Initializes an L-shaped Voronoi cell of a fixed size for testing the
 * convexity robustness. */
void voronoicell::init_l_shape() {
	for(int i=0;i<current_vertex_order;i++) mec[i]=0;
	up=0;
	mec[3]=p=12;
	const double j=0;
	*pts=-2;pts[1]=-2;pts[2]=-2;
	pts[4]=2;pts[5]=-2;pts[6]=-2;
	pts[8]=-2;pts[9]=0;pts[10]=-2;
	pts[12]=-j;pts[13]=j;pts[14]=-2;
	pts[16]=0;pts[17]=2;pts[18]=-2;
	pts[20]=2;pts[21]=2;pts[22]=-2;
	pts[24]=-2;pts[25]=-2;pts[26]=2;
	pts[28]=2;pts[29]=-2;pts[30]=2;
	pts[32]=-2;pts[33]=0;pts[34]=2;
	pts[36]=-j;pts[37]=j;pts[38]=2;
	pts[40]=0;pts[41]=2;pts[42]=2;
	pts[44]=2;pts[45]=2;pts[46]=2;
	int *q=mep[3];
	*q=1;q[1]=6;q[2]=2;q[6]=0;
	q[7]=5;q[8]=7;q[9]=0;q[13]=1;
	q[14]=0;q[15]=8;q[16]=3;q[20]=2;
	q[21]=2;q[22]=9;q[23]=4;q[27]=3;
	q[28]=3;q[29]=10;q[30]=5;q[34]=4;
	q[35]=4;q[36]=11;q[37]=1;q[41]=5;
	q[42]=8;q[43]=0;q[44]=7;q[48]=6;
	q[49]=6;q[50]=1;q[51]=11;q[55]=7;
	q[56]=9;q[57]=2;q[58]=6;q[62]=8;
	q[63]=10;q[64]=3;q[65]=8;q[69]=9;
	q[70]=11;q[71]=4;q[72]=9;q[76]=10;
	q[77]=7;q[78]=5;q[79]=10;q[83]=11;
	*ed=q;ed[1]=q+7;ed[2]=q+14;ed[3]=q+21;ed[4]=q+28;ed[5]=q+35;
	ed[6]=q+42;ed[7]=q+49;ed[8]=q+56;ed[9]=q+63;ed[10]=q+70;ed[11]=q+77;
	for(int i=0;i<12;i++) nu[i]=3;
	construct_relations();
}

/** Initializes a Voronoi cell as a regular octahedron.
 * \param[in] l The distance from the octahedron center to a vertex. Six
 *              vertices are initialized at (-l,0,0), (l,0,0), (0,-l,0),
 *              (0,l,0), (0,0,-l), and (0,0,l). */
void voronoicell_base::init_octahedron_base(double l) {
	for(int i=0;i<current_vertex_order;i++) mec[i]=0;
	up=0;
	mec[4]=p=6;l*=2;
	*pts=-l;pts[1]=0;pts[2]=0;
	pts[4]=l;pts[5]=0;pts[6]=0;
	pts[8]=0;pts[9]=-l;pts[10]=0;
	pts[12]=0;pts[13]=l;pts[14]=0;
	pts[16]=0;pts[17]=0;pts[18]=-l;
	pts[20]=0;pts[21]=0;pts[22]=l;
	int *q=mep[4];
	*q=2;q[1]=5;q[2]=3;q[3]=4;q[4]=0;q[5]=0;q[6]=0;q[7]=0;q[8]=0;
	q[9]=2;q[10]=4;q[11]=3;q[12]=5;q[13]=2;q[14]=2;q[15]=2;q[16]=2;q[17]=1;
	q[18]=0;q[19]=4;q[20]=1;q[21]=5;q[22]=0;q[23]=3;q[24]=0;q[25]=1;q[26]=2;
	q[27]=0;q[28]=5;q[29]=1;q[30]=4;q[31]=2;q[32]=3;q[33]=2;q[34]=1;q[35]=3;
	q[36]=0;q[37]=3;q[38]=1;q[39]=2;q[40]=3;q[41]=3;q[42]=1;q[43]=1;q[44]=4;
	q[45]=0;q[46]=2;q[47]=1;q[48]=3;q[49]=1;q[50]=3;q[51]=3;q[52]=1;q[53]=5;
	*ed=q;ed[1]=q+9;ed[2]=q+18;ed[3]=q+27;ed[4]=q+36;ed[5]=q+45;
	*nu=nu[1]=nu[2]=nu[3]=nu[4]=nu[5]=4;
}

/** Initializes a Voronoi cell as a tetrahedron. It assumes that the normal to
 * the face for the first three vertices points inside.
 * \param (x0,y0,z0) a position vector for the first vertex.
 * \param (x1,y1,z1) a position vector for the second vertex.
 * \param (x2,y2,z2) a position vector for the third vertex.
 * \param (x3,y3,z3) a position vector for the fourth vertex. */
void voronoicell_base::init_tetrahedron_base(double x0,double y0,double z0,double x1,double y1,double z1,double x2,double y2,double z2,double x3,double y3,double z3) {
	for(int i=0;i<current_vertex_order;i++) mec[i]=0;
	up=0;
	mec[3]=p=4;
	*pts=x0*2;pts[1]=y0*2;pts[2]=z0*2;
	pts[4]=x1*2;pts[5]=y1*2;pts[6]=z1*2;
	pts[8]=x2*2;pts[9]=y2*2;pts[10]=z2*2;
	pts[12]=x3*2;pts[13]=y3*2;pts[14]=z3*2;
	int *q=mep[3];
	*q=1;q[1]=3;q[2]=2;q[3]=0;q[4]=0;q[5]=0;q[6]=0;
	q[7]=0;q[8]=2;q[9]=3;q[10]=0;q[11]=2;q[12]=1;q[13]=1;
	q[14]=0;q[15]=3;q[16]=1;q[17]=2;q[18]=2;q[19]=1;q[20]=2;
	q[21]=0;q[22]=1;q[23]=2;q[24]=1;q[25]=2;q[26]=1;q[27]=3;
	*ed=q;ed[1]=q+7;ed[2]=q+14;ed[3]=q+21;
	*nu=nu[1]=nu[2]=nu[3]=3;
}

/** Checks that the relational table of the Voronoi cell is accurate, and
 * prints out any errors. This algorithm is O(p), so running it every time the
 * plane routine is called will result in a significant slowdown. */
void voronoicell_base::check_relations() {
	int i,j;
	for(i=0;i<p;i++) for(j=0;j<nu[i];j++) if(ed[ed[i][j]][ed[i][nu[i]+j]]!=i)
		printf("Relational error at point %d, edge %d.\n",i,j);
}

/** This routine checks for any two vertices that are connected by more than
 * one edge. The plane algorithm is designed so that this should not happen, so
 * any occurrences are most likely errors. Note that the routine is O(p), so
 * running it every time the plane routine is called will result in a
 * significant slowdown. */
void voronoicell_base::check_duplicates() {
	int i,j,k;
	for(i=0;i<p;i++) for(j=1;j<nu[i];j++) for(k=0;k<j;k++) if(ed[i][j]==ed[i][k])
		printf("Duplicate edges: (%d,%d) and (%d,%d) [%d]\n",i,j,i,k,ed[i][j]);
}

/** Constructs the relational table if the edges have been specified. */
void voronoicell_base::construct_relations() {
	int i,j,k,l;
	for(i=0;i<p;i++) for(j=0;j<nu[i];j++) {
		k=ed[i][j];
		l=0;
		while(ed[k][l]!=i) {
			l++;
			if(l==nu[k]) voro_fatal_error("Relation table construction failed",VOROPP_INTERNAL_ERROR);
		}
		ed[i][nu[i]+j]=l;
	}
}

/** Starting from a point within the current cutting plane, this routine attempts
 * to find an edge to a point outside the cutting plane. This prevents the plane
 * routine from .
 * \param[in,out] up */
inline bool voronoicell_base::search_for_outside_edge(int &up) {
	int i,lp,lw,*j=stackp2,sc2=stackp2-ds2;
	double l;
	*(stackp2++)=up;
	while(j<stackp2) {
		up=*(j++);
		for(i=0;i<nu[up];i++) {
			lp=ed[up][i];
			lw=m_test(lp,l);
			if(lw==0) {
				stackp2=ds2+sc2;
				return true;
			}
			else if(lw==1) add_to_stack(sc2,lp);
		}
	}
	stackp2=ds2+sc2;
	return false;
}

/** Adds a point to the auxiliary delete stack if it is not already there.
 * \param[in] vc a reference to the specialized version of the calling class.
 * \param[in] lp the index of the point to add.
 * \param[in,out] stackp2 a pointer to the end of the stack entries. */
inline void voronoicell_base::add_to_stack(int sc2,int lp) {
	for(int *k=ds2+sc2;k<stackp2;k++) if(*k==lp) return;
	if(stackp2==stacke2) add_memory_ds2();
	*(stackp2++)=lp;
}

/** Assuming that the point up is outside the cutting plane, this routine
 * searches upwards along edges trying to find an edge that intersects the
 * cutting plane.
 * \param[in] rsq the distance along this vector of the plane.
 * \param[in,out] u the dot product of point up with the normal.
 * \return True if the cutting plane was reached, false otherwise. */
inline bool voronoicell_base::search_upward(unsigned int &uw,int &lp,int &ls,int &us,double &l,double &u) {
	int vs;
	lp=up;l=u;

	// The test point is outside of the cutting space
	for(ls=0;ls<nu[lp];ls++) {
		up=ed[lp][ls];
		uw=m_test(up,u);
		if(u>l) break;
	}
	if(ls==nu[lp]) if(definite_max(lp,ls,l,u,uw)) {
		up=lp;
		return false;
	}

	while(uw==0) {
		//if(++count>=p) failsafe_find(lp,ls,us,l,u);

		// Test all the neighbors of the current point
		// and find the one which is closest to the
		// plane
		vs=ed[lp][nu[lp]+ls];lp=up;l=u;
		for(ls=0;ls<nu[lp];ls++) {
			if(ls==vs) continue;
			up=ed[lp][ls];
			uw=m_test(up,u);
			if(u>l) break;
		}
		if(ls==nu[lp]&&definite_max(lp,ls,l,u,uw)) {
			up=lp;
			return false;
		}
	}
	us=ed[lp][nu[lp]+ls];
	return true;
}

/** Checks whether a particular point lp is a definite maximum, searching
 * through any possible minor non-convexities, for a better maximum.
 * \param[in] (x,y,z) the normal vector to the plane. */
bool voronoicell_base::definite_max(int &lp,int &ls,double &l,double &u,unsigned int &uw) {
	int tp=lp,ts,qp=0;
	unsigned int qw;
	double q;

	// Check to see whether point up is a well-defined maximum. Otherwise
	// any neighboring vertices of up that are marginal need to be
	// followed, to see if they lead to a better maximum.
	for(ts=0;ts<nu[tp];ts++) {
		qp=ed[tp][ts];
		m_test(qp,q);
		if(q>l-big_tol) break;
	}
	if(ts==nu[tp]) return true;

	// The point tp is marginal, so it will be necessary to do the
	// flood-fill search. Mark the point tp and the point qp, and search
	// any remaining neighbors of the point tp.
	int *stackp=ds+1;
	flip(lp);
	flip(qp);
	*ds=qp;
	ts++;
	while(ts<nu[tp]) {
		qp=ed[tp][ts];
		m_test(qp,q);
		if(q>l-big_tol) {
			if(stackp==stacke) add_memory_ds();
			*(stackp++)=up;
			flip(up);
		}
		ts++;
	}

	// Consider additional marginal points, starting with the original
	// point qp
	int *spp=ds;
	while(spp<stackp) {
		tp=*(spp++);
		for(ts=0;ts<nu[tp];ts++) {
			qp=ed[tp][ts];

			// Skip the point if it's already marked
			if(ed[qp][nu[qp]<<1]<0) continue;
			qw=m_test(qp,q);

			// This point is a better maximum. Reset markers and
			// return true.
			if(q>l) {
				flip(lp);
				lp=tp;
				ls=ts;
				m_test(lp,l);
				up=qp;
				uw=qw;
				u=q;
				while(stackp>ds) flip(*(--stackp));
				return false;
			}

			// The point is marginal and therefore must also be
			// considered
			if(q>l-big_tol) {
				if(stackp==stacke) {
					int nn=stackp-spp;
					add_memory_ds();
					spp=stackp-nn;
				}
				*(stackp++)=qp;
				flip(qp);
			}
		}
	}

	// Reset markers and return false
	flip(lp);
	while(stackp>ds) flip(*(--stackp));
	return true;
}

inline bool voronoicell_base::search_downward(unsigned int &lw,int &lp,int &ls,int &us,double &l,double &u) {
	int vs;

	// The test point is outside of the cutting space
	for(us=0;us<nu[up];us++) {
		lp=ed[up][us];
		lw=m_test(lp,l);
		if(u>l) break;
	}
	if(us==nu[up]) if(definite_min(lp,us,l,u,lw)) return false;

	while(lw==2) {
		//if(++count>=p) failsafe_find(lp,ls,us,l,u);

		// Test all the neighbors of the current point
		// and find the one which is closest to the
		// plane
		vs=ed[up][nu[up]+us];up=lp;u=l;
		for(us=0;us<nu[up];us++) {
			if(us==vs) continue;
			lp=ed[up][us];
			lw=m_test(lp,l);
			if(u>l) break;
		}
		if(us==nu[up]&&definite_min(lp,us,l,u,lw)) return false;
	}
	ls=ed[up][nu[up]+us];
	return true;
}

bool voronoicell_base::definite_min(int &lp,int &us,double &l,double &u,unsigned int &lw) {
	int tp=up,ts,qp=0;
	unsigned int qw;
	double q;

	// Check to see whether point up is a well-defined maximum. Otherwise
	// any neighboring vertices of up that are marginal need to be
	// followed, to see if they lead to a better maximum.
	for(ts=0;ts<nu[tp];ts++) {
		qp=ed[tp][ts];
		m_test(qp,q);
		if(q<u+big_tol) break;
	}
	if(ts==nu[tp]) return true;

	// The point tp is marginal, so it will be necessary to do the
	// flood-fill search. Mark the point tp and the point qp, and search
	// any remaining neighbors of the point tp.
	int *stackp=ds+1;
	flip(up);
	flip(qp);
	*ds=qp;
	ts++;
	while(ts<nu[tp]) {
		qp=ed[tp][ts];
		m_test(qp,q);
		if(q<u+big_tol) {
			if(stackp==stacke) add_memory_ds();
			*(stackp++)=lp;
			flip(lp);
		}
		ts++;
	}

	// Consider additional marginal points, starting with the original
	// point qp
	int *spp=ds;
	while(spp<stackp) {
		tp=*(spp++);
		for(ts=0;ts<nu[tp];ts++) {
			qp=ed[tp][ts];

			// Skip the point if it's already marked
			if(ed[qp][nu[qp]<<1]<0) continue;
			qw=m_test(qp,q);

			// This point is a better minimum. Reset markers and
			// return true.
			if(q<u) {
				flip(up);
				up=tp;
				us=ts;
				m_test(up,u);
				lp=qp;
				lw=qw;
				l=q;
				while(stackp>ds) flip(*(--stackp));
				return false;
			}

			// The point is marginal and therefore must also be
			// considered
			if(q<u+big_tol) {
				if(stackp==stacke) {
					int nn=stackp-spp;
					add_memory_ds();
					spp=stackp-nn;
				}
				*(stackp++)=qp;
				flip(qp);
			}
		}
	}

	// Reset markers and return false
	flip(up);
	while(stackp>ds) flip(*(--stackp));
	return true;
}

/** Cuts the Voronoi cell by a particle whose center is at a separation of
 * (x,y,z) from the cell center. The value of rsq should be initially set to
 * \f$x^2+y^2+z^2\f$.
 * \param[in] vc a reference to the specialized version of the calling class.
 * \param[in] (x,y,z) the normal vector to the plane.
 * \param[in] rsq the distance along this vector of the plane.
 * \param[in] p_id the plane ID (for neighbor tracking only).
 * \return False if the plane cut deleted the cell entirely, true otherwise. */
template<class vc_class>
bool voronoicell_base::nplane(vc_class &vc,double x,double y,double z,double rsq,int p_id) {
	int i,j,lp=up,cp,qp,*dsp;
	int us=0,ls=0;
	unsigned int uw,lw;
	int *edp,*edd;stackp=ds;
	double u,l=0;up=0;

	// Initialize the safe testing routine
	px=x;py=y;pz=z;prsq=rsq;
	maskc+=4;
	if(maskc<4) reset_mask();

	uw=m_test(up,u);
	if(uw==2) {
		if(!search_downward(lw,lp,ls,us,l,u)) return false;
		if(lw==1) {up=lp;lp=-1;}
	} else if(uw==0) {
		if(!search_upward(uw,lp,ls,us,l,u)) return true;
		if(uw==1) lp=-1;
	} else {
		lp=-1;
	}

	// Set stack pointers
	stackp=ds;stackp2=ds2;stackp3=xse;

	// Store initial number of vertices
	int op=p;

	if(create_facet(vc,lp,ls,l,us,u,p_id)) return false;
	int k=0;int xtra=0;
	while(xse+k<stackp3) {
		lp=xse[k++];
		uw=m_test(lp,l);
		for(ls=0;ls<nu[lp];ls++) {
			up=ed[lp][ls];

			// Skip if this is a new vertex
			uw=m_test(up,u);
			if(up>=op) continue;

			if(uw==0) {
				if(u>-big_tol&&ed[up][nu[up]<<1]!=-1) {
					ed[up][nu[up]<<1]=-1;
					if(stackp3==stacke3) add_memory_xse();
					*(stackp3++)=up;
				}
			} else if(uw==1) {

				// This is a possible facet starting
				// from a vertex on the cutting plane
				if(create_facet(vc,-1,0,0,0,u,p_id)) return false;
			} else {

				// This is a new facet
				us=ed[lp][nu[lp]+ls];
				m_test(lp,l);
				if(create_facet(vc,lp,ls,l,us,u,p_id)) return false;
			}
		}
		xtra++;
	}

	// Reset back pointers on extra search stack
	for(dsp=xse;dsp<stackp3;dsp++) {
		j=*dsp;
		ed[j][nu[j]<<1]=j;
	}

	// Delete points: first, remove any duplicates
	dsp=ds;
	while(dsp<stackp) {
		j=*dsp;
		if(ed[j][nu[j]]!=-1) {
			ed[j][nu[j]]=-1;
			dsp++;
		} else *dsp=*(--stackp);
	}

	// Add the points in the auxiliary delete stack,
	// and reset their back pointers
	for(dsp=ds2;dsp<stackp2;dsp++) {
		j=*dsp;
		ed[j][nu[j]<<1]=j;
		if(ed[j][nu[j]]!=-1) {
			ed[j][nu[j]]=-1;
			if(stackp==stacke) add_memory_ds();
			*(stackp++)=j;
		}
	}

	// Scan connections and add in extras
	for(dsp=ds;dsp<stackp;dsp++) {
		cp=*dsp;
		for(edp=ed[cp];edp<ed[cp]+nu[cp];edp++) {
			qp=*edp;
			if(qp!=-1&&ed[qp][nu[qp]]!=-1) {
				if(stackp==stacke) {
					int dis=stackp-dsp;
					add_memory_ds();
					dsp=ds+dis;
				}
				*(stackp++)=qp;
				ed[qp][nu[qp]]=-1;
			}
		}
	}
	up=0;

	// Delete them from the array structure
	while(stackp>ds) {
		--p;
		while(ed[p][nu[p]]==-1) {
			j=nu[p];
			edp=ed[p];edd=(mep[j]+((j<<1)+1)*--mec[j]);
			while(edp<ed[p]+(j<<1)+1) *(edp++)=*(edd++);
			vc.n_set_aux2_copy(p,j);
			vc.n_copy_pointer(ed[p][(j<<1)],p);
			ed[ed[p][(j<<1)]]=ed[p];
			--p;
		}
		up=*(--stackp);
		if(up<p) {

			// Vertex management
			pts[(up<<2)]=pts[(p<<2)];
			pts[(up<<2)+1]=pts[(p<<2)+1];
			pts[(up<<2)+2]=pts[(p<<2)+2];

			// Memory management
			j=nu[up];
			edp=ed[up];edd=(mep[j]+((j<<1)+1)*--mec[j]);
			while(edp<ed[up]+(j<<1)+1) *(edp++)=*(edd++);
			vc.n_set_aux2_copy(up,j);
			vc.n_copy_pointer(ed[up][j<<1],up);
			vc.n_copy_pointer(up,p);
			ed[ed[up][j<<1]]=ed[up];

			// Edge management
			ed[up]=ed[p];
			nu[up]=nu[p];
			for(i=0;i<nu[up];i++) ed[ed[up][i]][ed[up][nu[up]+i]]=up;
			ed[up][nu[up]<<1]=up;
		} else up=p++;
	}

	// Check for any vertices of zero order
	if(*mec>0) voro_fatal_error("Zero order vertex formed",VOROPP_INTERNAL_ERROR);

	// Collapse any order 2 vertices and exit
	return collapse_order2(vc);
}

/** Creates a new facet.
 * \return True if cell deleted, false otherwise. */
template<class vc_class>
bool voronoicell_base::create_facet(vc_class &vc,int lp,int ls,double l,int us,double u,int p_id) {
	int i,j,k,qp,qs,iqs,cp,cs,rp,*edp,*edd;
	unsigned int lw,qw;
	bool new_double_edge=false,double_edge=false;
	double q,r;

	// We're about to add the first point of the new facet. In either
	// routine, we have to add a point, so first check there's space for
	// it.
	if(p==current_vertices) add_memory_vertices(vc);

	if(lp==-1) {

		// We want to be strict about reaching the conclusion that the
		// cell is entirely within the cutting plane. It's not enough
		// to find a vertex that has edges which are all inside or on
		// the plane. If the vertex has neighbors that are also on the
		// plane, we should check those too.
		if(!search_for_outside_edge(up)) return true;

		// The search algorithm found a point which is on the cutting
		// plane. We leave that point in place, and create a new one at
		// the same location.
		pts[(p<<2)]=pts[(up<<2)];
		pts[(p<<2)+1]=pts[(up<<2)+1];
		pts[(p<<2)+2]=pts[(up<<2)+2];

		// Search for a collection of edges of the test vertex which
		// are outside of the cutting space. Begin by testing the
		// zeroth edge.
		i=0;
		lp=*ed[up];
		lw=m_testx(lp,l);
		if(lw!=0) {

			// The first edge is either inside the cutting space,
			// or lies within the cutting plane. Test the edges
			// sequentially until we find one that is outside.
			unsigned int rw=lw;
			do {
				i++;

				// If we reached the last edge with no luck
				// then all of the vertices are inside
				// or on the plane, so the cell is completely
				// deleted
				if(i==nu[up]) return true;
				lp=ed[up][i];
				lw=m_testx(lp,l);
			} while (lw!=0);
			j=i+1;

			// We found an edge outside the cutting space. Keep
			// moving through these edges until we find one that's
			// inside or on the plane.
			while(j<nu[up]) {
				lp=ed[up][j];
				lw=m_testx(lp,l);
				if(lw!=0) break;
				j++;
			}

			// Compute the number of edges for the new vertex. In
			// general it will be the number of outside edges
			// found, plus two. But we need to recognize the
			// special case when all but one edge is outside, and
			// the remaining one is on the plane. For that case we
			// have to reduce the edge count by one to prevent
			// doubling up.
			if(j==nu[up]&&i==1&&rw==1) {
				nu[p]=nu[up];
				double_edge=true;
			} else nu[p]=j-i+2;
			k=1;

			// Add memory for the new vertex if needed, and
			// initialize
			while (nu[p]>=current_vertex_order) add_memory_vorder(vc);
			if(mec[nu[p]]==mem[nu[p]]) add_memory(vc,nu[p]);
			vc.n_set_pointer(p,nu[p]);
			ed[p]=mep[nu[p]]+((nu[p]<<1)+1)*mec[nu[p]]++;
			ed[p][nu[p]<<1]=p;

			// Copy the edges of the original vertex into the new
			// one. Delete the edges of the original vertex, and
			// update the relational table.
			us=cycle_down(i,up);
			while(i<j) {
				qp=ed[up][i];
				qs=ed[up][nu[up]+i];
				vc.n_copy(p,k,up,i);
				ed[p][k]=qp;
				ed[p][nu[p]+k]=qs;
				ed[qp][qs]=p;
				ed[qp][nu[qp]+qs]=k;
				ed[up][i]=-1;
				i++;k++;
			}
			qs=i==nu[up]?0:i;
		} else {

			// In this case, the zeroth edge is outside the cutting
			// plane. Begin by searching backwards from the last
			// edge until we find an edge which isn't outside.
			i=nu[up]-1;
			lp=ed[up][i];
			lw=m_testx(lp,l);
			while(lw==0) {
				i--;

				// If i reaches zero, then we have a point in
				// the plane all of whose edges are outside
				// the cutting space, so we just exit
				if(i==0) return false;
				lp=ed[up][i];
				lw=m_testx(lp,l);
			}

			// Now search forwards from zero
			j=1;
			qp=ed[up][j];
			qw=m_testx(qp,q);
			while(qw==0) {
				j++;
				qp=ed[up][j];
				qw=m_testx(qp,l);
			}

			// Compute the number of edges for the new vertex. In
			// general it will be the number of outside edges
			// found, plus two. But we need to recognize the
			// special case when all but one edge is outside, and
			// the remaining one is on the plane. For that case we
			// have to reduce the edge count by one to prevent
			// doubling up.
			if(i==j&&qw==1) {
				double_edge=true;
				nu[p]=nu[up];
			} else {
				nu[p]=nu[up]-i+j+1;
			}

			// Add memory to store the vertex if it doesn't exist
			// already
			k=1;
			while(nu[p]>=current_vertex_order) add_memory_vorder(vc);
			if(mec[nu[p]]==mem[nu[p]]) add_memory(vc,nu[p]);

			// Copy the edges of the original vertex into the new
			// one. Delete the edges of the original vertex, and
			// update the relational table.
			vc.n_set_pointer(p,nu[p]);
			ed[p]=mep[nu[p]]+((nu[p]<<1)+1)*mec[nu[p]]++;
			ed[p][nu[p]<<1]=p;
			us=i++;
			while(i<nu[up]) {
				qp=ed[up][i];
				qs=ed[up][nu[up]+i];
				vc.n_copy(p,k,up,i);
				ed[p][k]=qp;
				ed[p][nu[p]+k]=qs;
				ed[qp][qs]=p;
				ed[qp][nu[qp]+qs]=k;
				ed[up][i]=-1;
				i++;k++;
			}
			i=0;
			while(i<j) {
				qp=ed[up][i];
				qs=ed[up][nu[up]+i];
				vc.n_copy(p,k,up,i);
				ed[p][k]=qp;
				ed[p][nu[p]+k]=qs;
				ed[qp][qs]=p;
				ed[qp][nu[qp]+qs]=k;
				ed[up][i]=-1;
				i++;k++;
			}
			qs=j;
		}
		if(!double_edge) {
			vc.n_copy(p,k,up,qs);
			vc.n_set(p,0,p_id);
		} else vc.n_copy(p,0,up,qs);

		// Add this point to the auxiliary delete stack
		if(stackp2==stacke2) add_memory_ds2();
		*(stackp2++)=up;

		// Look at the edges on either side of the group that was
		// detected. We're going to commence facet computation by
		// moving along one of them. We are going to end up coming back
		// along the other one.
		cs=k;
		qp=up;q=u;
		i=ed[up][us];
		us=ed[up][nu[up]+us];
		up=i;
		ed[qp][nu[qp]<<1]=-p;

	} else {

		// The search algorithm found an intersected edge between the
		// points lp and up. Create a new vertex between them which
		// lies on the cutting plane. Since u and l differ by at least
		// the tolerance, this division should never screw up.
		if(stackp==stacke) add_memory_ds();
		*(stackp++)=up;
		r=u/(u-l);l=1-r;
		pts[p<<2]=pts[lp<<2]*r+pts[up<<2]*l;
		pts[(p<<2)+1]=pts[(lp<<2)+1]*r+pts[(up<<2)+1]*l;
		pts[(p<<2)+2]=pts[(lp<<2)+2]*r+pts[(up<<2)+2]*l;

		// This point will always have three edges. Connect one of them
		// to lp.
		nu[p]=3;
		if(mec[3]==mem[3]) add_memory(vc,3);
		vc.n_set_pointer(p,3);
		vc.n_set(p,0,p_id);
		vc.n_copy(p,1,up,us);
		vc.n_copy(p,2,lp,ls);
		ed[p]=mep[3]+7*mec[3]++;
		ed[p][6]=p;
		ed[up][us]=-1;
		ed[lp][ls]=p;
		ed[lp][nu[lp]+ls]=1;
		ed[p][1]=lp;
		ed[p][nu[p]+1]=ls;
		cs=2;

		// Set the direction to move in
		qs=cycle_up(us,up);
		qp=up;q=u;
	}

	// When the code reaches here, we have initialized the first point, and
	// we have a direction for moving it to construct the rest of the facet
	cp=p;rp=p;p++;
	while(qp!=up||qs!=us) {

		// We're currently tracing round an intersected facet. Keep
		// moving around it until we find a point or edge which
		// intersects the plane.
		lp=ed[qp][qs];
		lw=m_testx(lp,l);

		if(lw==2) {

			// The point is still in the cutting space. Just add it
			// to the delete stack and keep moving.
			qs=cycle_up(ed[qp][nu[qp]+qs],lp);
			qp=lp;
			q=l;
			if(stackp==stacke) add_memory_ds();
			*(stackp++)=qp;

		} else if(lw==0) {

			// The point is outside of the cutting space, so we've
			// found an intersected edge. Introduce a regular point
			// at the point of intersection. Connect it to the
			// point we just tested. Also connect it to the previous
			// new point in the facet we're constructing.
			if(p==current_vertices) add_memory_vertices(vc);
			r=q/(q-l);l=1-r;
			pts[p<<2]=pts[lp<<2]*r+pts[qp<<2]*l;
			pts[(p<<2)+1]=pts[(lp<<2)+1]*r+pts[(qp<<2)+1]*l;
			pts[(p<<2)+2]=pts[(lp<<2)+2]*r+pts[(qp<<2)+2]*l;
			nu[p]=3;
			if(mec[3]==mem[3]) add_memory(vc,3);
			ls=ed[qp][qs+nu[qp]];
			vc.n_set_pointer(p,3);
			vc.n_set(p,0,p_id);
			vc.n_copy(p,1,qp,qs);
			vc.n_copy(p,2,lp,ls);
			ed[p]=mep[3]+7*mec[3]++;
			*ed[p]=cp;
			ed[p][1]=lp;
			ed[p][3]=cs;
			ed[p][4]=ls;
			ed[p][6]=p;
			ed[lp][ls]=p;
			ed[lp][nu[lp]+ls]=1;
			ed[cp][cs]=p;
			ed[cp][nu[cp]+cs]=0;
			ed[qp][qs]=-1;
			qs=cycle_up(qs,qp);
			cp=p++;
			cs=2;
		} else {

			// We've found a point which is on the cutting plane.
			// We're going to introduce a new point right here, but
			// first we need to figure out the number of edges it
			// has.
			if(p==current_vertices) add_memory_vertices(vc);

			// If the previous vertex detected a double edge, our
			// new vertex will have one less edge.
			k=double_edge?0:1;
			qs=ed[qp][nu[qp]+qs];
			qp=lp;
			iqs=qs;

			// Start testing the edges of the current point until
			// we find one which isn't outside the cutting space
			do {
				k++;
				qs=cycle_up(qs,qp);
				lp=ed[qp][qs];
				lw=m_testx(lp,l);
			} while (lw==0);

			// Now we need to find out whether this marginal vertex
			// we are on has been visited before, because if that's
			// the case, we need to add vertices to the existing
			// new vertex, rather than creating a fresh one. We also
			// need to figure out whether we're in a case where we
			// might be creating a duplicate edge.
			j=-ed[qp][nu[qp]<<1];
	 		if(qp==up&&qs==us) {

				// If we're heading into the final part of the
				// new facet, then we never worry about the
				// duplicate edge calculation.
				new_double_edge=false;
				if(j>0) k+=nu[j];
			} else {
				if(j>0) {

					// This vertex was visited before, so
					// count those vertices to the ones we
					// already have.
					k+=nu[j];

					// The only time when we might make a
					// duplicate edge is if the point we're
					// going to move to next is also a
					// marginal point, so test for that
					// first.
					if(lw==1) {

						// Now see whether this marginal point
						// has been visited before.
						i=-ed[lp][nu[lp]<<1];
						if(i>0) {

							// Now see if the last edge of that other
							// marginal point actually ends up here.
							if(ed[i][nu[i]-1]==j) {
								new_double_edge=true;
								k-=1;
							} else new_double_edge=false;
						} else {

							// That marginal point hasn't been visited
							// before, so we probably don't have to worry
							// about duplicate edges, except in the
							// case when that's the way into the end
							// of the facet, because that way always creates
							// an edge.
							if(j==rp&&lp==up&&ed[qp][nu[qp]+qs]==us) {
								new_double_edge=true;
								k-=1;
							} else new_double_edge=false;
						}
					} else new_double_edge=false;
				} else {

					// The vertex hasn't been visited
					// before, but let's see if it's
					// marginal
					if(lw==1) {

						// If it is, we need to check
						// for the case that it's a
						// small branch, and that we're
						// heading right back to where
						// we came from
						i=-ed[lp][nu[lp]<<1];
						if(i==cp) {
							new_double_edge=true;
							k-=1;
						} else new_double_edge=false;
					} else new_double_edge=false;
				}
			}

			// k now holds the number of edges of the new vertex
			// we are forming. Add memory for it if it doesn't exist
			// already.
			while(k>=current_vertex_order) add_memory_vorder(vc);
			if(mec[k]==mem[k]) add_memory(vc,k);

			// Now create a new vertex with order k, or augment
			// the existing one
			if(j>0) {

				// If we're augmenting a vertex but we don't
				// actually need any more edges, just skip this
				// routine to avoid memory confusion
				if(nu[j]!=k) {

					// Allocate memory and copy the edges
					// of the previous instance into it
					vc.n_set_aux1(k);
					edp=mep[k]+((k<<1)+1)*mec[k]++;
					i=0;
					while(i<nu[j]) {
						vc.n_copy_aux1(j,i);
						edp[i]=ed[j][i];
						edp[k+i]=ed[j][nu[j]+i];
						i++;
					}
					edp[k<<1]=j;

					// Remove the previous instance with
					// fewer vertices from the memory
					// structure
					edd=mep[nu[j]]+((nu[j]<<1)+1)*--mec[nu[j]];
					if(edd!=ed[j]) {
						for(int lll=0;lll<=(nu[j]<<1);lll++) ed[j][lll]=edd[lll];
						vc.n_set_aux2_copy(j,nu[j]);
						vc.n_copy_pointer(edd[nu[j]<<1],j);
						ed[edd[nu[j]<<1]]=ed[j];
					}
					vc.n_set_to_aux1(j);
					ed[j]=edp;
				} else i=nu[j];
			} else {

				// Allocate a new vertex of order k
				vc.n_set_pointer(p,k);
				ed[p]=mep[k]+((k<<1)+1)*mec[k]++;
				ed[p][k<<1]=p;
				if(stackp2==stacke2) add_memory_ds2();
				*(stackp2++)=qp;
				pts[p<<2]=pts[qp<<2];
				pts[(p<<2)+1]=pts[(qp<<2)+1];
				pts[(p<<2)+2]=pts[(qp<<2)+2];
				ed[qp][nu[qp]<<1]=-p;
				j=p++;
				i=0;
			}
			nu[j]=k;

			// Unless the previous case was a double edge, connect
			// the first available edge of the new vertex to the
			// last one in the facet
			if(!double_edge) {
				ed[j][i]=cp;
				ed[j][nu[j]+i]=cs;
				vc.n_set(j,i,p_id);
				ed[cp][cs]=j;
				ed[cp][nu[cp]+cs]=i;
				i++;
			}

			// Copy in the edges of the underlying vertex,
			// and do one less if this was a double edge
			qs=iqs;
			while(i<(new_double_edge?k:k-1)) {
				qs=cycle_up(qs,qp);
				lp=ed[qp][qs];ls=ed[qp][nu[qp]+qs];
				vc.n_copy(j,i,qp,qs);
				ed[j][i]=lp;
				ed[j][nu[j]+i]=ls;
				ed[lp][ls]=j;
				ed[lp][nu[lp]+ls]=i;
				ed[qp][qs]=-1;
				i++;
			}
			qs=cycle_up(qs,qp);
			cs=i;
			cp=j;
			vc.n_copy(j,new_double_edge?0:cs,qp,qs);

			// Update the double_edge flag, to pass it
			// to the next instance of this routine
			double_edge=new_double_edge;
		}
	}

	// Connect the final created vertex to the initial one
	ed[cp][cs]=rp;
	*ed[rp]=cp;
	ed[cp][nu[cp]+cs]=0;
	ed[rp][nu[rp]]=cs;
	return false;
}

/** During the creation of a new facet in the plane routine, it is possible
 * that some order two vertices may arise. This routine removes them.
 * Suppose an order two vertex joins c and d. If there's a edge between
 * c and d already, then the order two vertex is just removed; otherwise,
 * the order two vertex is removed and c and d are joined together directly.
 * It is possible this process will create order two or order one vertices,
 * and the routine is continually run until all of them are removed.
 * \return False if the vertex removal was unsuccessful, indicative of the cell
 *         reducing to zero volume and disappearing; true if the vertex removal
 *         was successful. */
template<class vc_class>
inline bool voronoicell_base::collapse_order2(vc_class &vc) {
	if(!collapse_order1(vc)) return false;
	int a,b,i,j,k,l;
	while(mec[2]>0) {

		// Pick a order 2 vertex and read in its edges
		i=--mec[2];
		j=mep[2][5*i];k=mep[2][5*i+1];
		if(j==k) {
#if VOROPP_VERBOSE >=1
			fputs("Order two vertex joins itself",stderr);
#endif
			return false;
		}

		// Scan the edges of j to see if joins k
		for(l=0;l<nu[j];l++) {
			if(ed[j][l]==k) break;
		}

		// If j doesn't already join k, join them together.
		// Otherwise delete the connection to the current
		// vertex from j and k.
		a=mep[2][5*i+2];b=mep[2][5*i+3];i=mep[2][5*i+4];
		if(l==nu[j]) {
			ed[j][a]=k;
			ed[k][b]=j;
			ed[j][nu[j]+a]=b;
			ed[k][nu[k]+b]=a;
		} else {
			if(!delete_connection(vc,j,a,false)) return false;
			if(!delete_connection(vc,k,b,true)) return false;
		}

		// Compact the memory
		--p;
		if(up==i) up=0;
		if(p!=i) {
			if(up==p) up=i;
			pts[i<<2]=pts[p<<2];
			pts[(i<<2)+1]=pts[(p<<2)+1];
			pts[(i<<2)+2]=pts[(p<<2)+2];
			for(k=0;k<nu[p];k++) ed[ed[p][k]][ed[p][nu[p]+k]]=i;
			vc.n_copy_pointer(i,p);
			ed[i]=ed[p];
			nu[i]=nu[p];
			ed[i][nu[i]<<1]=i;
		}

		// Collapse any order 1 vertices if they were created
		if(!collapse_order1(vc)) return false;
	}
	return true;
}

/** Order one vertices can potentially be created during the order two collapse
 * routine. This routine keeps removing them until there are none left.
 * \return False if the vertex removal was unsuccessful, indicative of the cell
 *         having zero volume and disappearing; true if the vertex removal was
 *         successful. */
template<class vc_class>
bool voronoicell_base::collapse_order1(vc_class &vc) {
	int i,j,k;
	while(mec[1]>0) {
		up=0;
#if VOROPP_VERBOSE >=1
		fputs("Order one collapse\n",stderr);
#endif
		i=--mec[1];
		j=mep[1][3*i];k=mep[1][3*i+1];
		i=mep[1][3*i+2];
		if(!delete_connection(vc,j,k,false)) return false;
		--p;
		if(up==i) up=0;
		if(p!=i) {
			if(up==p) up=i;
			pts[i<<2]=pts[p<<2];
			pts[(i<<2)+1]=pts[(p<<2)+1];
			pts[(i<<2)+2]=pts[(p<<2)+2];
			for(k=0;k<nu[p];k++) ed[ed[p][k]][ed[p][nu[p]+k]]=i;
			vc.n_copy_pointer(i,p);
			ed[i]=ed[p];
			nu[i]=nu[p];
			ed[i][nu[i]<<1]=i;
		}
	}
	return true;
}

/** This routine deletes the kth edge of vertex j and reorganizes the memory.
 * If the neighbor computation is enabled, we also have to supply an handedness
 * flag to decide whether to preserve the plane on the left or right of the
 * connection.
 * \return False if a zero order vertex was formed, indicative of the cell
 *         disappearing; true if the vertex removal was successful. */
template<class vc_class>
bool voronoicell_base::delete_connection(vc_class &vc,int j,int k,bool hand) {
	int q=hand?k:cycle_up(k,j);
	int i=nu[j]-1,l,*edp,*edd,m;
#if VOROPP_VERBOSE >=1
	if(i<1) {
		fputs("Zero order vertex formed\n",stderr);
		return false;
	}
#endif
	if(mec[i]==mem[i]) add_memory(vc,i);
	vc.n_set_aux1(i);
	for(l=0;l<q;l++) vc.n_copy_aux1(j,l);
	while(l<i) {
		vc.n_copy_aux1_shift(j,l);
		l++;
	}
	edp=mep[i]+((i<<1)+1)*mec[i]++;
	edp[i<<1]=j;
	for(l=0;l<k;l++) {
		edp[l]=ed[j][l];
		edp[l+i]=ed[j][l+nu[j]];
	}
	while(l<i) {
		m=ed[j][l+1];
		edp[l]=m;
		k=ed[j][l+nu[j]+1];
		edp[l+i]=k;
		ed[m][nu[m]+k]--;
		l++;
	}

	edd=mep[nu[j]]+((nu[j]<<1)+1)*--mec[nu[j]];
	for(l=0;l<=(nu[j]<<1);l++) ed[j][l]=edd[l];
	vc.n_set_aux2_copy(j,nu[j]);
	vc.n_copy_pointer(edd[nu[j]<<1],j);
	vc.n_set_to_aux1(j);
	ed[edd[nu[j]<<1]]=ed[j];
	ed[j]=edp;
	nu[j]=i;
	return true;
}

/** This routine is a fall-back, in case floating point errors caused the usual
 * search routine to fail. In the fall-back routine, we just test every edge to
 * find one straddling the plane. */
bool voronoicell_base::failsafe_find(int &lp,int &ls,int &us,double &l,double &u) {
	fputs("Bailed out of convex calculation (not supported yet)\n",stderr);
	exit(1);
/*	qw=1;lw=0;
	for(qp=0;qp<p;qp++) {
		qw=m_test(qp,q);
		if(qw==1) {

			// The point is inside the cutting space. Now
			// see if we can find a neighbor which isn't.
			for(us=0;us<nu[qp];us++) {
				lp=ed[qp][us];
				if(lp<qp) {
					lw=m_test(lp,l);
					if(lw!=1) break;
				}
			}
			if(us<nu[qp]) {
				up=qp;
				if(lw==0) {
					complicated_setup=true;
				} else {
					complicated_setup=false;
					u=q;
					ls=ed[up][nu[up]+us];
				}
				break;
			}
		} else if(qw==-1) {

			// The point is outside the cutting space. See
			// if we can find a neighbor which isn't.
			for(ls=0;ls<nu[qp];ls++) {
				up=ed[qp][ls];
				if(up<qp) {
					uw=m_test(up,u);
					if(uw!=-1) break;
				}
			}
			if(ls<nu[qp]) {
				if(uw==0) {
					up=qp;
					complicated_setup=true;
				} else {
					complicated_setup=false;
					lp=qp;l=q;
					us=ed[lp][nu[lp]+ls];
				}
				break;
			}
		} else {

			// The point is in the plane, so we just
			// proceed with the complicated setup routine
			up=qp;
			complicated_setup=true;
			break;
		}
	}
	if(qp==p) return qw==-1?true:false;*/
}

/** Calculates the volume of the Voronoi cell, by decomposing the cell into
 * tetrahedra extending outward from the zeroth vertex, whose volumes are
 * evaluated using a scalar triple product.
 * \return A floating point number holding the calculated volume. */
double voronoicell_base::volume() {
	const double fe=1/48.0;
	double vol=0;
	int i,j,k,l,m,n;
	double ux,uy,uz,vx,vy,vz,wx,wy,wz;
	for(i=1;i<p;i++) {
		ux=*pts-pts[i<<2];
		uy=pts[1]-pts[(i<<2)+1];
		uz=pts[2]-pts[(i<<2)+2];
		for(j=0;j<nu[i];j++) {
			k=ed[i][j];
			if(k>=0) {
				ed[i][j]=-1-k;
				l=cycle_up(ed[i][nu[i]+j],k);
				vx=pts[k<<2]-*pts;
				vy=pts[(k<<2)+1]-pts[1];
				vz=pts[(k<<2)+2]-pts[2];
				m=ed[k][l];ed[k][l]=-1-m;
				while(m!=i) {
					n=cycle_up(ed[k][nu[k]+l],m);
					wx=pts[(m<<2)]-*pts;
					wy=pts[(m<<2)+1]-pts[1];
					wz=pts[(m<<2)+2]-pts[2];
					vol+=ux*vy*wz+uy*vz*wx+uz*vx*wy-uz*vy*wx-uy*vx*wz-ux*vz*wy;
					k=m;l=n;vx=wx;vy=wy;vz=wz;
					m=ed[k][l];ed[k][l]=-1-m;
				}
			}
		}
	}
	reset_edges();
	return vol*fe;
}

/** Calculates the contributions to the Minkowski functionals for this Voronoi cell.
 * \param[in] r the radius to consider.
 * \param[out] ar the area functional.
 * \param[out] vo the volume functional. */
void voronoicell_base::minkowski(double r,double &ar,double &vo) {
	int i,j,k,l,m,n;
	ar=vo=0;r*=2;
	for(i=1;i<p;i++) for(j=0;j<nu[i];j++) {
		k=ed[i][j];
		if(k>=0) {
			ed[i][j]=-1-k;
			l=cycle_up(ed[i][nu[i]+j],k);
			m=ed[k][l];ed[k][l]=-1-m;
			while(m!=i) {
				n=cycle_up(ed[k][nu[k]+l],m);
				minkowski_contrib(i,k,m,r,ar,vo);
				k=m;l=n;
				m=ed[k][l];ed[k][l]=-1-m;
			}
		}
	}
	vo*=0.125;
	ar*=0.25;
	reset_edges();
}

inline void voronoicell_base::minkowski_contrib(int i,int k,int m,double r,double &ar,double &vo) {
	double ix=pts[4*i],iy=pts[4*i+1],iz=pts[4*i+2],
	       kx=pts[4*k],ky=pts[4*k+1],kz=pts[4*k+2],
	       mx=pts[4*m],my=pts[4*m+1],mz=pts[4*m+2],
	       ux=kx-ix,uy=ky-iy,uz=kz-iz,vx=mx-kx,vy=my-ky,vz=mz-kz,
	       e1x=uz*vy-uy*vz,e1y=ux*vz-uz*vx,e1z=uy*vx-ux*vy,e2x,e2y,e2z,
	       wmag=e1x*e1x+e1y*e1y+e1z*e1z;
	if(wmag<tol*tol) return;
	wmag=1/sqrt(wmag);
	e1x*=wmag;e1y*=wmag;e1z*=wmag;

	// Compute second orthonormal vector
	if(fabs(e1x)>0.5) {
		e2x=-e1y;e2y=e1x;e2z=0;
	} else if(fabs(e1y)>0.5) {
		e2x=0;e2y=-e1z;e2z=e1y;
	} else {
		e2x=e1z;e2y=0;e2z=-e1x;
	}
	wmag=1/sqrt(e2x*e2x+e2y*e2y+e2z*e2z);
	e2x*=wmag;e2y*=wmag;e2z*=wmag;

	// Compute third orthonormal vector
	double e3x=e1z*e2y-e1y*e2z,
	       e3y=e1x*e2z-e1z*e2x,
	       e3z=e1y*e2x-e1x*e2y,
	       x0=e1x*ix+e1y*iy+e1z*iz;
	if(x0<tol) return;

	double ir=e2x*ix+e2y*iy+e2z*iz,is=e3x*ix+e3y*iy+e3z*iz,
	       kr=e2x*kx+e2y*ky+e2z*kz,ks=e3x*kx+e3y*ky+e3z*kz,
	       mr=e2x*mx+e2y*my+e2z*mz,ms=e3x*mx+e3y*my+e3z*mz;

	minkowski_edge(x0,ir,is,kr,ks,r,ar,vo);
	minkowski_edge(x0,kr,ks,mr,ms,r,ar,vo);
	minkowski_edge(x0,mr,ms,ir,is,r,ar,vo);
}

void voronoicell_base::minkowski_edge(double x0,double r1,double s1,double r2,double s2,double r,double &ar,double &vo) {
	double r12=r2-r1,s12=s2-s1,l12=r12*r12+s12*s12;
	if(l12<tol*tol) return;
	l12=1/sqrt(l12);r12*=l12;s12*=l12;
	double y0=s12*r1-r12*s1;
	if(fabs(y0)<tol) return;
	minkowski_formula(x0,y0,-r12*r1-s12*s1,r,ar,vo);
	minkowski_formula(x0,y0,r12*r2+s12*s2,r,ar,vo);
}

void voronoicell_base::minkowski_formula(double x0,double y0,double z0,double r,double &ar,double &vo) {
	const double pi=3.1415926535897932384626433832795;
	if(fabs(z0)<tol) return;
	double si;
	if(z0<0) {z0=-z0;si=-1;} else si=1;
	if(y0<0) {y0=-y0;si=-si;}
	double xs=x0*x0,ys=y0*y0,zs=z0*z0,res=xs+ys,rvs=res+zs,theta=atan(z0/y0),rs=r*r,rc=rs*r,temp,voc,arc;
	if(r<x0) {
		temp=2*theta-0.5*pi-asin((zs*xs-ys*rvs)/(res*(ys+zs)));
		voc=rc/6.*temp;
		arc=rs*0.5*temp;
	} else if(rs<res*1.0000000001) {
		temp=0.5*pi+asin((zs*xs-ys*rvs)/(res*(ys+zs)));
		voc=theta*0.5*(rs*x0-xs*x0/3.)-rc/6.*temp;
		arc=theta*x0*r-rs*0.5*temp;
	} else if(rs<rvs) {
		temp=theta-pi*0.5+asin(y0/sqrt(rs-xs));
		double temp2=(rs*x0-xs*x0/3.),
		       x2s=rs*xs/res,y2s=rs*ys/res,
		       temp3=asin((x2s-y2s-xs)/(rs-xs)),
		       temp4=asin((zs*xs-ys*rvs)/(res*(ys+zs))),
		       temp5=sqrt(rs-res);
		voc=0.5*temp*temp2+x0*y0/6.*temp5+r*rs/6*(temp3-temp4);
		arc=x0*r*temp-0.5*temp2*y0*r/((rs-xs)*temp5)+x0*y0/6.*r/temp5+rs*0.5*temp3+rs*rs/3.*2*xs*ys/(res*(rs-xs)*sqrt((rs-xs)*(rs-xs)-(x2s-y2s-xs)*(x2s-y2s-xs)))-rs*0.5*temp4;
	} else {
		voc=x0*y0*z0/6.;
		arc=0;
	}
	vo+=voc*si;
	ar+=arc*si;
}

static double dot_product(double *a, double *b) {
	return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

static void cross_product(double *a, double *b, double *c) {
	c[0]=a[1]*b[2]-a[2]*b[1];
	c[1]=a[2]*b[0]-a[0]*b[2];
	c[2]=a[0]*b[1]-a[1]*b[0];
}

static double triple_product(double *a, double *b, double *c) {
	double cp[3];
	cross_product(b,c,cp);
	return dot_product(a,cp);
}

static void normalize_vector(double *x, double *normalized) {
	double normsq = dot_product(x,x);
	double invnorm = normsq<=0?0:1/sqrt(normsq);
	for (int i=0;i<3;i++) {
		normalized[i]=x[i]*invnorm;
	}
}

static double calculate_solid_angle(double *R1, double *R2, double *R3) {
	// Method described in:
	// A. Van Oosterom and J. Strackee
	// "The Solid Angle of a Plane Triangle"
	// IEEE Transactions on Biomedical Engineering, BME-30, 2, 1983, 125--126
	// https://doi.org/10.1109/TBME.1983.325207
	return fabs(2*atan2(triple_product(R1,R2,R3),1+dot_product(R1,R2)+dot_product(R2,R3)+dot_product(R3,R1)));
}

/** Calculates the solid angles of each face of the Voronoi cell and prints
 * the results to an output stream.
 * \param[out] v the vector to store the results in. */
void voronoicell_base::solid_angles(std::vector<double> &v) {
	double solid_angle;
	std::vector<double> normalized(3*p);
	v.clear();
	int i,j,k,l,m,n;
	for (i=0;i<p;i++) {
		normalize_vector(&pts[4*i], &normalized[3*i]);
	}

	for(i=1;i<p;i++) for(j=0;j<nu[i];j++) {
		k=ed[i][j];
		if(k>=0) {
			solid_angle=0;
			ed[i][j]=-1-k;
			l=cycle_up(ed[i][nu[i]+j],k);
			m=ed[k][l];ed[k][l]=-1-m;
			while(m!=i) {
				n=cycle_up(ed[k][nu[k]+l],m);
				solid_angle+=calculate_solid_angle(&normalized[3*i],&normalized[3*k],&normalized[3*m]);
				k=m;l=n;
				m=ed[k][l];ed[k][l]=-1-m;
			}
			v.push_back(solid_angle);
		}
	}
	reset_edges();
}

/** Calculates the areas of each face of the Voronoi cell and prints the
 * results to an output stream.
 * \param[out] v the vector to store the results in. */
void voronoicell_base::face_areas(std::vector<double> &v) {
	double area;
	v.clear();
	int i,j,k,l,m,n;
	double ux,uy,uz,vx,vy,vz,wx,wy,wz;
	for(i=1;i<p;i++) for(j=0;j<nu[i];j++) {
		k=ed[i][j];
		if(k>=0) {
			area=0;
			ed[i][j]=-1-k;
			l=cycle_up(ed[i][nu[i]+j],k);
			m=ed[k][l];ed[k][l]=-1-m;
			while(m!=i) {
				n=cycle_up(ed[k][nu[k]+l],m);
				ux=pts[4*k]-pts[4*i];
				uy=pts[4*k+1]-pts[4*i+1];
				uz=pts[4*k+2]-pts[4*i+2];
				vx=pts[4*m]-pts[4*i];
				vy=pts[4*m+1]-pts[4*i+1];
				vz=pts[4*m+2]-pts[4*i+2];
				wx=uy*vz-uz*vy;
				wy=uz*vx-ux*vz;
				wz=ux*vy-uy*vx;
				area+=sqrt(wx*wx+wy*wy+wz*wz);
				k=m;l=n;
				m=ed[k][l];ed[k][l]=-1-m;
			}
			v.push_back(0.125*area);
		}
	}
	reset_edges();
}

/** Calculates the total surface area of the Voronoi cell.
 * \return The computed area. */
double voronoicell_base::surface_area() {
	double area=0;
	int i,j,k,l,m,n;
	double ux,uy,uz,vx,vy,vz,wx,wy,wz;
	for(i=1;i<p;i++) for(j=0;j<nu[i];j++) {
		k=ed[i][j];
		if(k>=0) {
			ed[i][j]=-1-k;
			l=cycle_up(ed[i][nu[i]+j],k);
			m=ed[k][l];ed[k][l]=-1-m;
			while(m!=i) {
				n=cycle_up(ed[k][nu[k]+l],m);
				ux=pts[4*k]-pts[4*i];
				uy=pts[4*k+1]-pts[4*i+1];
				uz=pts[4*k+2]-pts[4*i+2];
				vx=pts[4*m]-pts[4*i];
				vy=pts[4*m+1]-pts[4*i+1];
				vz=pts[4*m+2]-pts[4*i+2];
				wx=uy*vz-uz*vy;
				wy=uz*vx-ux*vz;
				wz=ux*vy-uy*vx;
				area+=sqrt(wx*wx+wy*wy+wz*wz);
				k=m;l=n;
				m=ed[k][l];ed[k][l]=-1-m;
			}
		}
	}
	reset_edges();
	return 0.125*area;
}

/** Calculates the centroid of the Voronoi cell, by decomposing the cell into
 * tetrahedra extending outward from the zeroth vertex.
 * \param[out] (cx,cy,cz) references to floating point numbers in which to
 *                        pass back the centroid vector. */
void voronoicell_base::centroid(double &cx,double &cy,double &cz) {
	double tvol,vol=0;cx=cy=cz=0;
	int i,j,k,l,m,n;
	double ux,uy,uz,vx,vy,vz,wx,wy,wz;
	for(i=1;i<p;i++) {
		ux=*pts-pts[4*i];
		uy=pts[1]-pts[4*i+1];
		uz=pts[2]-pts[4*i+2];
		for(j=0;j<nu[i];j++) {
			k=ed[i][j];
			if(k>=0) {
				ed[i][j]=-1-k;
				l=cycle_up(ed[i][nu[i]+j],k);
				vx=pts[4*k]-*pts;
				vy=pts[4*k+1]-pts[1];
				vz=pts[4*k+2]-pts[2];
				m=ed[k][l];ed[k][l]=-1-m;
				while(m!=i) {
					n=cycle_up(ed[k][nu[k]+l],m);
					wx=pts[4*m]-*pts;
					wy=pts[4*m+1]-pts[1];
					wz=pts[4*m+2]-pts[2];
					tvol=ux*vy*wz+uy*vz*wx+uz*vx*wy-uz*vy*wx-uy*vx*wz-ux*vz*wy;
					vol+=tvol;
					cx+=(wx+vx-ux)*tvol;
					cy+=(wy+vy-uy)*tvol;
					cz+=(wz+vz-uz)*tvol;
					k=m;l=n;vx=wx;vy=wy;vz=wz;
					m=ed[k][l];ed[k][l]=-1-m;
				}
			}
		}
	}
	reset_edges();
	if(vol>tol_cu) {
		vol=0.125/vol;
		cx=cx*vol+0.5*(*pts);
		cy=cy*vol+0.5*pts[1];
		cz=cz*vol+0.5*pts[2];
	} else cx=cy=cz=0;
}

/** Computes the maximum radius squared of a vertex from the center of the
 * cell. It can be used to determine when enough particles have been testing an
 * all planes that could cut the cell have been considered.
 * \return The maximum radius squared of a vertex.*/
double voronoicell_base::max_radius_squared() {
	double r,s,*ptsp=pts+4,*ptse=pts+(p<<2);
	r=*pts*(*pts)+pts[1]*pts[1]+pts[2]*pts[2];
	while(ptsp<ptse) {
		s=*ptsp*(*ptsp);ptsp++;
		s+=*ptsp*(*ptsp);ptsp++;
		s+=*ptsp*(*ptsp);ptsp+=2;
		if(s>r) r=s;
	}
	return r;
}

/** Calculates the total edge distance of the Voronoi cell.
 * \return A floating point number holding the calculated distance. */
double voronoicell_base::total_edge_distance() {
	int i,j,k;
	double dis=0,dx,dy,dz;
	for(i=0;i<p-1;i++) for(j=0;j<nu[i];j++) {
		k=ed[i][j];
		if(k>i) {
			dx=pts[k<<2]-pts[i<<2];
			dy=pts[(k<<2)+1]-pts[(i<<2)+1];
			dz=pts[(k<<2)+2]-pts[(i<<2)+2];
			dis+=sqrt(dx*dx+dy*dy+dz*dz);
		}
	}
	return 0.5*dis;
}

/** Outputs the edges of the Voronoi cell in POV-Ray format to an open file
 * stream, displacing the cell by given vector.
 * \param[in] (x,y,z) a displacement vector to be added to the cell's position.
 * \param[in] fp a file handle to write to. */
void voronoicell_base::draw_pov(double x,double y,double z,FILE* fp) {
	int i,j,k;double *ptsp=pts,*pt2;
	char posbuf1[128],posbuf2[128];
	for(i=0;i<p;i++,ptsp+=4) {
		sprintf(posbuf1,"%g,%g,%g",x+*ptsp*0.5,y+ptsp[1]*0.5,z+ptsp[2]*0.5);
		fprintf(fp,"sphere{<%s>,r}\n",posbuf1);
		for(j=0;j<nu[i];j++) {
			k=ed[i][j];
			if(k<i) {
				pt2=pts+(k<<2);
				sprintf(posbuf2,"%g,%g,%g",x+*pt2*0.5,y+0.5*pt2[1],z+0.5*pt2[2]);
				if(strcmp(posbuf1,posbuf2)!=0) fprintf(fp,"cylinder{<%s>,<%s>,r}\n",posbuf1,posbuf2);
			}
		}
	}
}

/** Outputs the edges of the Voronoi cell in gnuplot format to an output stream.
 * \param[in] (x,y,z) a displacement vector to be added to the cell's position.
 * \param[in] fp a file handle to write to. */
void voronoicell_base::draw_gnuplot(double x,double y,double z,FILE *fp) {
	int i,j,k,l,m;
	for(i=1;i<p;i++) for(j=0;j<nu[i];j++) {
		k=ed[i][j];
		if(k>=0) {
			fprintf(fp,"%g %g %g\n",x+0.5*pts[i<<2],y+0.5*pts[(i<<2)+1],z+0.5*pts[(i<<2)+2]);
			l=i;m=j;
			do {
				ed[k][ed[l][nu[l]+m]]=-1-l;
				ed[l][m]=-1-k;
				l=k;
				fprintf(fp,"%g %g %g\n",x+0.5*pts[k<<2],y+0.5*pts[(k<<2)+1],z+0.5*pts[(k<<2)+2]);
			} while (search_edge(l,m,k));
			fputs("\n\n",fp);
		}
	}
	reset_edges();
}

inline bool voronoicell_base::search_edge(int l,int &m,int &k) {
	for(m=0;m<nu[l];m++) {
		k=ed[l][m];
		if(k>=0) return true;
	}
	return false;
}

/** Outputs the Voronoi cell in the POV mesh2 format, described in section
 * 1.3.2.2 of the POV-Ray documentation. The mesh2 output consists of a list of
 * vertex vectors, followed by a list of triangular faces. The routine also
 * makes use of the optional inside_vector specification, which makes the mesh
 * object solid, so that the POV-Ray Constructive Solid Geometry (CSG) can be
 * applied.
 * \param[in] (x,y,z) a displacement vector to be added to the cell's position.
 * \param[in] fp a file handle to write to. */
void voronoicell_base::draw_pov_mesh(double x,double y,double z,FILE *fp) {
	int i,j,k,l,m,n;
	double *ptsp=pts;
	fprintf(fp,"mesh2 {\nvertex_vectors {\n%d\n",p);
	for(i=0;i<p;i++,ptsp+=4) fprintf(fp,",<%g,%g,%g>\n",x+*ptsp*0.5,y+ptsp[1]*0.5,z+ptsp[2]*0.5);
	fprintf(fp,"}\nface_indices {\n%d\n",(p-2)<<1);
	for(i=1;i<p;i++) for(j=0;j<nu[i];j++) {
		k=ed[i][j];
		if(k>=0) {
			ed[i][j]=-1-k;
			l=cycle_up(ed[i][nu[i]+j],k);
			m=ed[k][l];ed[k][l]=-1-m;
			while(m!=i) {
				n=cycle_up(ed[k][nu[k]+l],m);
				fprintf(fp,",<%d,%d,%d>\n",i,k,m);
				k=m;l=n;
				m=ed[k][l];ed[k][l]=-1-m;
			}
		}
	}
	fputs("}\ninside_vector <0,0,1>\n}\n",fp);
	reset_edges();
}

/** Several routines in the class that gather cell-based statistics internally
 * track their progress by flipping edges to negative so that they know what
 * parts of the cell have already been tested. This function resets them back
 * to positive. When it is called, it assumes that every edge in the routine
 * should have already been flipped to negative, and it bails out with an
 * internal error if it encounters a positive edge. */
inline void voronoicell_base::reset_edges() {
	int i,j;
	for(i=0;i<p;i++) for(j=0;j<nu[i];j++) {
		if(ed[i][j]>=0) voro_fatal_error("Edge reset routine found a previously untested edge",VOROPP_INTERNAL_ERROR);
		ed[i][j]=-1-ed[i][j];
	}
}

/** Checks to see if a given vertex is inside, outside or within the test
 * plane. If the point is far away from the test plane, the routine immediately
 * returns whether it is inside or outside. If the routine is close the the
 * plane and within the specified tolerance, then the special check_marginal()
 * routine is called.
 * \param[in] n the vertex to test.
 * \param[out] ans the result of the scalar product used in evaluating the
 *                 location of the point.
 * \return -1 if the point is inside the plane, 1 if the point is outside the
 *         plane, or 0 if the point is within the plane. */
inline unsigned int voronoicell_base::m_test(int n,double &ans) {
	if(mask[n]>=maskc) {
		ans=pts[4*n+3];
		return mask[n]&3;
	} else return m_calc(n,ans);
}

unsigned int voronoicell_base::m_calc(int n,double &ans) {
	double *pp=pts+4*n;
	ans=*(pp++)*px;
	ans+=*(pp++)*py;
	ans+=*(pp++)*pz-prsq;
	*pp=ans;
	unsigned int maskr=ans<-tol?0:(ans>tol?2:1);
	mask[n]=maskc|maskr;
	return maskr;
}

/** Checks to see if a given vertex is inside, outside or within the test
 * plane. If the point is far away from the test plane, the routine immediately
 * returns whether it is inside or outside. If the routine is close the the
 * plane and within the specified tolerance, then the special check_marginal()
 * routine is called.
 * \param[in] n the vertex to test.
 * \param[out] ans the result of the scalar product used in evaluating the
 *                 location of the point.
 * \return -1 if the point is inside the plane, 1 if the point is outside the
 *         plane, or 0 if the point is within the plane. */
inline unsigned int voronoicell_base::m_testx(int n,double &ans) {
	unsigned int maskr;
	if(mask[n]>=maskc) {
		ans=pts[4*n+3];
		maskr=mask[n]&3;
	} else maskr=m_calc(n,ans);
	if(maskr==0&&ans>-big_tol&&ed[n][nu[n]<<1]!=-1) {
		ed[n][nu[n]<<1]=-1;
		if(stackp3==stacke3) add_memory_xse();
		*(stackp3++)=n;
	}
	return maskr;
}

/** This routine calculates the unit normal vectors for every face.
 * \param[out] v the vector to store the results in. */
void voronoicell_base::normals(std::vector<double> &v) {
	int i,j,k;
	v.clear();
	for(i=1;i<p;i++) for(j=0;j<nu[i];j++) {
		k=ed[i][j];
		if(k>=0) normals_search(v,i,j,k);
	}
	reset_edges();
}

/** This inline routine is called by normals(). It attempts to construct a
 * single normal vector that is associated with a particular face. It first
 * traces around the face, trying to find two vectors along the face edges
 * whose vector product is above the numerical tolerance. It then constructs
 * the normal vector using this product. If the face is too small, and none of
 * the vector products are large enough, the routine may return (0,0,0) as the
 * normal vector.
 * \param[in] v the vector to store the results in.
 * \param[in] i the initial vertex of the face to test.
 * \param[in] j the index of an edge of the vertex.
 * \param[in] k the neighboring vertex of i, set to ed[i][j]. */
inline void voronoicell_base::normals_search(std::vector<double> &v,int i,int j,int k) {
	ed[i][j]=-1-k;
	int l=cycle_up(ed[i][nu[i]+j],k),m;
	double ux,uy,uz,vx,vy,vz,wx,wy,wz,wmag;
	do {
		m=ed[k][l];ed[k][l]=-1-m;
		ux=pts[4*m]-pts[4*k];
		uy=pts[4*m+1]-pts[4*k+1];
		uz=pts[4*m+2]-pts[4*k+2];

		// Test to see if the length of this edge is above the tolerance
		if(ux*ux+uy*uy+uz*uz>tol) {
			while(m!=i) {
				l=cycle_up(ed[k][nu[k]+l],m);
				k=m;m=ed[k][l];ed[k][l]=-1-m;
				vx=pts[4*m]-pts[4*k];
				vy=pts[4*m+1]-pts[4*k+1];
				vz=pts[4*m+2]-pts[4*k+2];

				// Construct the vector product of this edge with
				// the previous one
				wx=uz*vy-uy*vz;
				wy=ux*vz-uz*vx;
				wz=uy*vx-ux*vy;
				wmag=wx*wx+wy*wy+wz*wz;

				// Test to see if this vector product of the
				// two edges is above the tolerance
				if(wmag>tol) {

					// Construct the normal vector and print it
					wmag=1/sqrt(wmag);
					v.push_back(wx*wmag);
					v.push_back(wy*wmag);
					v.push_back(wz*wmag);

					// Mark all of the remaining edges of this
					// face and exit
					while(m!=i) {
						l=cycle_up(ed[k][nu[k]+l],m);
						k=m;m=ed[k][l];ed[k][l]=-1-m;
					}
					return;
				}
			}
			v.push_back(0);
			v.push_back(0);
			v.push_back(0);
			return;
		}
		l=cycle_up(ed[k][nu[k]+l],m);
		k=m;
	} while (k!=i);
	v.push_back(0);
	v.push_back(0);
	v.push_back(0);
}

/** Returns the number of faces of a computed Voronoi cell.
 * \return The number of faces. */
int voronoicell_base::number_of_faces() {
	int i,j,k,l,m,s=0;
	for(i=1;i<p;i++) for(j=0;j<nu[i];j++) {
		k=ed[i][j];
		if(k>=0) {
			s++;
			ed[i][j]=-1-k;
			l=cycle_up(ed[i][nu[i]+j],k);
			do {
				m=ed[k][l];
				ed[k][l]=-1-m;
				l=cycle_up(ed[k][nu[k]+l],m);
				k=m;
			} while (k!=i);

		}
	}
	reset_edges();
	return s;
}

/** Returns a vector of the vertex orders.
 * \param[out] v the vector to store the results in. */
void voronoicell_base::vertex_orders(std::vector<int> &v) {
	v.resize(p);
	for(int i=0;i<p;i++) v[i]=nu[i];
}

/** Outputs the vertex orders.
 * \param[out] fp the file handle to write to. */
void voronoicell_base::output_vertex_orders(FILE *fp) {
	if(p>0) {
		fprintf(fp,"%d",*nu);
		for(int *nup=nu+1;nup<nu+p;nup++) fprintf(fp," %d",*nup);
	}
}

/** Returns a vector of the vertex vectors using the local coordinate system.
 * \param[out] v the vector to store the results in. */
void voronoicell_base::vertices(std::vector<double> &v) {
	v.resize(3*p);
	double *ptsp=pts;
	for(int i=0;i<3*p;i+=3) {
		v[i]=*(ptsp++)*0.5;
		v[i+1]=*(ptsp++)*0.5;
		v[i+2]=*ptsp*0.5;ptsp+=2;
	}
}

/** Outputs the vertex vectors using the local coordinate system.
 * \param[out] fp the file handle to write to. */
void voronoicell_base::output_vertices(FILE *fp) {
	if(p>0) {
		fprintf(fp,"(%g,%g,%g)",*pts*0.5,pts[1]*0.5,pts[2]*0.5);
		for(double *ptsp=pts+4;ptsp<pts+(p<<2);ptsp+=4) fprintf(fp," (%g,%g,%g)",*ptsp*0.5,ptsp[1]*0.5,ptsp[2]*0.5);
	}
}

/** Returns a vector of the vertex vectors in the global coordinate system.
 * \param[out] v the vector to store the results in.
 * \param[in] (x,y,z) the position vector of the particle in the global
 *                    coordinate system. */
void voronoicell_base::vertices(double x,double y,double z,std::vector<double> &v) {
	v.resize(3*p);
	double *ptsp=pts;
	for(int i=0;i<3*p;i+=3) {
		v[i]=x+*(ptsp++)*0.5;
		v[i+1]=y+*(ptsp++)*0.5;
		v[i+2]=z+*ptsp*0.5;ptsp+=2;
	}
}

/** Outputs the vertex vectors using the global coordinate system.
 * \param[out] fp the file handle to write to.
 * \param[in] (x,y,z) the position vector of the particle in the global
 *                    coordinate system. */
void voronoicell_base::output_vertices(double x,double y,double z,FILE *fp) {
	if(p>0) {
		fprintf(fp,"(%g,%g,%g)",x+*pts*0.5,y+pts[1]*0.5,z+pts[2]*0.5);
		for(double *ptsp=pts+4;ptsp<pts+(p<<2);ptsp+=4) fprintf(fp," (%g,%g,%g)",x+*ptsp*0.5,y+ptsp[1]*0.5,z+ptsp[2]*0.5);
	}
}

/** This routine returns the perimeters of each face.
 * \param[out] v the vector to store the results in. */
void voronoicell_base::face_perimeters(std::vector<double> &v) {
	v.clear();
	int i,j,k,l,m;
	double dx,dy,dz,perim;
	for(i=1;i<p;i++) for(j=0;j<nu[i];j++) {
		k=ed[i][j];
		if(k>=0) {
			dx=pts[k<<2]-pts[i<<2];
			dy=pts[(k<<2)+1]-pts[(i<<2)+1];
			dz=pts[(k<<2)+2]-pts[(i<<2)+2];
			perim=sqrt(dx*dx+dy*dy+dz*dz);
			ed[i][j]=-1-k;
			l=cycle_up(ed[i][nu[i]+j],k);
			do {
				m=ed[k][l];
				dx=pts[m<<2]-pts[k<<2];
				dy=pts[(m<<2)+1]-pts[(k<<2)+1];
				dz=pts[(m<<2)+2]-pts[(k<<2)+2];
				perim+=sqrt(dx*dx+dy*dy+dz*dz);
				ed[k][l]=-1-m;
				l=cycle_up(ed[k][nu[k]+l],m);
				k=m;
			} while (k!=i);
			v.push_back(0.5*perim);
		}
	}
	reset_edges();
}

/** For each face, this routine outputs a bracketed sequence of numbers
 * containing a list of all the vertices that make up that face.
 * \param[out] v the vector to store the results in. */
void voronoicell_base::face_vertices(std::vector<int> &v) {
	int i,j,k,l,m,vp(0),vn;
	v.clear();
	for(i=1;i<p;i++) for(j=0;j<nu[i];j++) {
		k=ed[i][j];
		if(k>=0) {
			v.push_back(0);
			v.push_back(i);
			ed[i][j]=-1-k;
			l=cycle_up(ed[i][nu[i]+j],k);
			do {
				v.push_back(k);
				m=ed[k][l];
				ed[k][l]=-1-m;
				l=cycle_up(ed[k][nu[k]+l],m);
				k=m;
			} while (k!=i);
			vn=v.size();
			v[vp]=vn-vp-1;
			vp=vn;
		}
	}
	reset_edges();
}

/** Outputs a list of the number of edges in each face.
 * \param[out] v the vector to store the results in. */
void voronoicell_base::face_orders(std::vector<int> &v) {
	int i,j,k,l,m,q;
	v.clear();
	for(i=1;i<p;i++) for(j=0;j<nu[i];j++) {
		k=ed[i][j];
		if(k>=0) {
			q=1;
			ed[i][j]=-1-k;
			l=cycle_up(ed[i][nu[i]+j],k);
			do {
				q++;
				m=ed[k][l];
				ed[k][l]=-1-m;
				l=cycle_up(ed[k][nu[k]+l],m);
				k=m;
			} while (k!=i);
			v.push_back(q);;
		}
	}
	reset_edges();
}

/** Computes the number of edges that each face has and outputs a frequency
 * table of the results.
 * \param[out] v the vector to store the results in. */
void voronoicell_base::face_freq_table(std::vector<int> &v) {
	int i,j,k,l,m,q;
	v.clear();
	for(i=1;i<p;i++) for(j=0;j<nu[i];j++) {
		k=ed[i][j];
		if(k>=0) {
			q=1;
			ed[i][j]=-1-k;
			l=cycle_up(ed[i][nu[i]+j],k);
			do {
				q++;
				m=ed[k][l];
				ed[k][l]=-1-m;
				l=cycle_up(ed[k][nu[k]+l],m);
				k=m;
			} while (k!=i);
			if((unsigned int) q>=v.size()) v.resize(q+1,0);
			v[q]++;
		}
	}
	reset_edges();
}

/** This routine tests to see whether the cell intersects a plane by starting
 * from the guess point up. If up intersects, then it immediately returns true.
 * Otherwise, it calls the plane_intersects_track() routine.
 * \param[in] (x,y,z) the normal vector to the plane.
 * \param[in] rsq the distance along this vector of the plane.
 * \return False if the plane does not intersect the plane, true if it does. */
bool voronoicell_base::plane_intersects(double x,double y,double z,double rsq) {
	double g=x*pts[up<<2]+y*pts[(up<<2)+1]+z*pts[(up<<2)+2];
	if(g<rsq) return plane_intersects_track(x,y,z,rsq,g);
	return true;
}

/** This routine tests to see if a cell intersects a plane. It first tests a
 * random sample of approximately sqrt(p)/4 points. If any of those are
 * intersect, then it immediately returns true. Otherwise, it takes the closest
 * point and passes that to plane_intersect_track() routine.
 * \param[in] (x,y,z) the normal vector to the plane.
 * \param[in] rsq the distance along this vector of the plane.
 * \return False if the plane does not intersect the plane, true if it does. */
bool voronoicell_base::plane_intersects_guess(double x,double y,double z,double rsq) {
	up=0;
	double g=x*pts[up<<2]+y*pts[(up<<2)+1]+z*pts[(up<<2)+2];
	if(g<rsq) {
		int ca=1,cc=p>>3,mp=1;
		double m;
		while(ca<cc) {
			m=x*pts[4*mp]+y*pts[4*mp+1]+z*pts[4*mp+2];
			if(m>g) {
				if(m>rsq) return true;
				g=m;up=mp;
			}
			ca+=mp++;
		}
		return plane_intersects_track(x,y,z,rsq,g);
	}
	return true;
}

/* This routine tests to see if a cell intersects a plane, by tracing over the
 * cell from vertex to vertex, starting at up. It is meant to be called either
 * by plane_intersects() or plane_intersects_track(), when those routines
 * cannot immediately resolve the case.
 * \param[in] (x,y,z) the normal vector to the plane.
 * \param[in] rsq the distance along this vector of the plane.
 * \param[in] g the distance of up from the plane.
 * \return False if the plane does not intersect the plane, true if it does. */
inline bool voronoicell_base::plane_intersects_track(double x,double y,double z,double rsq,double g) {

	for(int tp=0;tp<p;tp++) if(x*pts[tp<<2]+y*pts[(tp<<2)+1]+z*pts[(tp<<2)+2]>rsq) return true;
	return false;
/*
	int ls,us,lp;
	double l,u;
	unsigned int uw;

	// Initialize the safe testing routine
	px=x;py=y;pz=z;prsq=rsq;
	maskc+=4;
	if(maskc<4) reset_mask();

	return search_upward(uw,lp,ls,us,l,u);
}*/
	/*
	int count=0,ls,us,tp;
	double t;
	// The test point is outside of the cutting space
	for(us=0;us<nu[up];us++) {
		tp=ed[up][us];
		t=x*pts[tp<<2]+y*pts[(tp<<2)+1]+z*pts[(tp<<2)+2];
		if(t>g) {
			ls=ed[up][nu[up]+us];
			up=tp;
			while (t<rsq) {
				if(++count>=p) {
#if VOROPP_VERBOSE >=1
					fputs("Bailed out of convex calculation",stderr);
#endif
					for(tp=0;tp<p;tp++) if(x*pts[tp<<2]+y*pts[(tp<<2)+1]+z*pts[(tp<<2)+2]>rsq) return true;
					return false;
				}

				// Test all the neighbors of the current point
				// and find the one which is closest to the
				// plane
				for(us=0;us<ls;us++) {
					tp=ed[up][us];double *pp=pts+(tp<<2);
					g=x*(*pp)+y*pp[1]+z*pp[2];
					if(g>t) break;
				}
				if(us==ls) {
					us++;
					while(us<nu[up]) {
						tp=ed[up][us];double *pp=pts+(tp<<2);
						g=x*(*pp)+y*pp[1]+z*pp[2];
						if(g>t) break;
						us++;
					}
					if(us==nu[up]) return false;
				}
				ls=ed[up][nu[up]+us];up=tp;t=g;
			}
			return true;
		}
	}
	return false;*/
}

/** Counts the number of edges of the Voronoi cell.
 * \return the number of edges. */
int voronoicell_base::number_of_edges() {
	int edges=0,*nup=nu;
	while(nup<nu+p) edges+=*(nup++);
	return edges>>1;
}

/** Outputs a custom string of information about the Voronoi cell. The string
 * of information follows a similar style as the C printf command, and detailed
 * information about its format is available at
 * http://math.lbl.gov/voro++/doc/custom.html.
 * \param[in] format the custom string to print.
 * \param[in] i the ID of the particle associated with this Voronoi cell.
 * \param[in] (x,y,z) the position of the particle associated with this Voronoi
 *                    cell.
 * \param[in] r a radius associated with the particle.
 * \param[in] fp the file handle to write to. */
void voronoicell_base::output_custom(const char *format,int i,double x,double y,double z,double r,FILE *fp) {
	char *fmp=(const_cast<char*>(format));
	std::vector<int> vi;
	std::vector<double> vd;
	while(*fmp!=0) {
		if(*fmp=='%') {
			fmp++;
			switch(*fmp) {

				// Particle-related output
				case 'i': fprintf(fp,"%d",i);break;
				case 'x': fprintf(fp,"%g",x);break;
				case 'y': fprintf(fp,"%g",y);break;
				case 'z': fprintf(fp,"%g",z);break;
				case 'q': fprintf(fp,"%g %g %g",x,y,z);break;
				case 'r': fprintf(fp,"%g",r);break;

				// Vertex-related output
				case 'w': fprintf(fp,"%d",p);break;
				case 'p': output_vertices(fp);break;
				case 'P': output_vertices(x,y,z,fp);break;
				case 'o': output_vertex_orders(fp);break;
				case 'm': fprintf(fp,"%g",0.25*max_radius_squared());break;

				// Edge-related output
				case 'g': fprintf(fp,"%d",number_of_edges());break;
				case 'E': fprintf(fp,"%g",total_edge_distance());break;
				case 'e': face_perimeters(vd);voro_print_vector(vd,fp);break;

				// Face-related output
				case 's': fprintf(fp,"%d",number_of_faces());break;
				case 'F': fprintf(fp,"%g",surface_area());break;
				case 'A': {
						  face_freq_table(vi);
						  voro_print_vector(vi,fp);
					  } break;
				case 'a': face_orders(vi);voro_print_vector(vi,fp);break;
				case 'f': face_areas(vd);voro_print_vector(vd,fp);break;
				case 't': {
						  face_vertices(vi);
						  voro_print_face_vertices(vi,fp);
					  } break;
				case 'l': normals(vd);
					  voro_print_positions(vd,fp);
					  break;
				case 'n': neighbors(vi);
					  voro_print_vector(vi,fp);
					  break;

				// Volume-related output
				case 'v': fprintf(fp,"%g",volume());break;
				case 'c': {
						  double cx,cy,cz;
						  centroid(cx,cy,cz);
						  fprintf(fp,"%g %g %g",cx,cy,cz);
					  } break;
				case 'C': {
						  double cx,cy,cz;
						  centroid(cx,cy,cz);
						  fprintf(fp,"%g %g %g",x+cx,y+cy,z+cz);
					  } break;

				// End-of-string reached
				case 0: fmp--;break;

				// The percent sign is not part of a
				// control sequence
				default: putc('%',fp);putc(*fmp,fp);
			}
		} else putc(*fmp,fp);
		fmp++;
	}
	fputs("\n",fp);
}

/** This initializes the class to be a rectangular box. It calls the base class
 * initialization routine to set up the edge and vertex information, and then
 * sets up the neighbor information, with initial faces being assigned ID
 * numbers from -1 to -6.
 * \param[in] (xmin,xmax) the minimum and maximum x coordinates.
 * \param[in] (ymin,ymax) the minimum and maximum y coordinates.
 * \param[in] (zmin,zmax) the minimum and maximum z coordinates. */
void voronoicell_neighbor::init(double xmin,double xmax,double ymin,double ymax,double zmin,double zmax) {
	init_base(xmin,xmax,ymin,ymax,zmin,zmax);
	int *q=mne[3];
	*q=-5;q[1]=-3;q[2]=-1;
	q[3]=-5;q[4]=-2;q[5]=-3;
	q[6]=-5;q[7]=-1;q[8]=-4;
	q[9]=-5;q[10]=-4;q[11]=-2;
	q[12]=-6;q[13]=-1;q[14]=-3;
	q[15]=-6;q[16]=-3;q[17]=-2;
	q[18]=-6;q[19]=-4;q[20]=-1;
	q[21]=-6;q[22]=-2;q[23]=-4;
	*ne=q;ne[1]=q+3;ne[2]=q+6;ne[3]=q+9;
	ne[4]=q+12;ne[5]=q+15;ne[6]=q+18;ne[7]=q+21;
}

/** This initializes the class to be an octahedron. It calls the base class
 * initialization routine to set up the edge and vertex information, and then
 * sets up the neighbor information, with the initial faces being assigned ID
 * numbers from -1 to -8.
 * \param[in] l The distance from the octahedron center to a vertex. Six
 *              vertices are initialized at (-l,0,0), (l,0,0), (0,-l,0),
 *              (0,l,0), (0,0,-l), and (0,0,l). */
void voronoicell_neighbor::init_octahedron(double l) {
	init_octahedron_base(l);
	int *q=mne[4];
	*q=-5;q[1]=-6;q[2]=-7;q[3]=-8;
	q[4]=-1;q[5]=-2;q[6]=-3;q[7]=-4;
	q[8]=-6;q[9]=-5;q[10]=-2;q[11]=-1;
	q[12]=-8;q[13]=-7;q[14]=-4;q[15]=-3;
	q[16]=-5;q[17]=-8;q[18]=-3;q[19]=-2;
	q[20]=-7;q[21]=-6;q[22]=-1;q[23]=-4;
	*ne=q;ne[1]=q+4;ne[2]=q+8;ne[3]=q+12;ne[4]=q+16;ne[5]=q+20;
}

/** This initializes the class to be a tetrahedron. It calls the base class
 * initialization routine to set up the edge and vertex information, and then
 * sets up the neighbor information, with the initial faces being assigned ID
 * numbers from -1 to -4.
 * \param (x0,y0,z0) a position vector for the first vertex.
 * \param (x1,y1,z1) a position vector for the second vertex.
 * \param (x2,y2,z2) a position vector for the third vertex.
 * \param (x3,y3,z3) a position vector for the fourth vertex. */
void voronoicell_neighbor::init_tetrahedron(double x0,double y0,double z0,double x1,double y1,double z1,double x2,double y2,double z2,double x3,double y3,double z3) {
	init_tetrahedron_base(x0,y0,z0,x1,y1,z1,x2,y2,z2,x3,y3,z3);
	int *q=mne[3];
	*q=-4;q[1]=-3;q[2]=-2;
	q[3]=-3;q[4]=-4;q[5]=-1;
	q[6]=-4;q[7]=-2;q[8]=-1;
	q[9]=-2;q[10]=-3;q[11]=-1;
	*ne=q;ne[1]=q+3;ne[2]=q+6;ne[3]=q+9;
}

/** This routine checks to make sure the neighbor information of each face is
 * consistent. */
void voronoicell_neighbor::check_facets() {
	int i,j,k,l,m,q;
	for(i=1;i<p;i++) for(j=0;j<nu[i];j++) {
		k=ed[i][j];
		if(k>=0) {
			ed[i][j]=-1-k;
			q=ne[i][j];
			l=cycle_up(ed[i][nu[i]+j],k);
			do {
				m=ed[k][l];
				ed[k][l]=-1-m;
				if(ne[k][l]!=q) fprintf(stderr,"Facet error at (%d,%d)=%d, started from (%d,%d)=%d\n",k,l,ne[k][l],i,j,q);
				l=cycle_up(ed[k][nu[k]+l],m);
				k=m;
			} while (k!=i);
		}
	}
	reset_edges();
}

/** The class constructor allocates memory for storing neighbor information. */
void voronoicell_neighbor::memory_setup() {
	int i;
	mne=new int*[current_vertex_order];
	ne=new int*[current_vertices];
	for(i=0;i<3;i++) mne[i]=new int[init_n_vertices*i];
	mne[3]=new int[init_3_vertices*3];
	for(i=4;i<current_vertex_order;i++) mne[i]=new int[init_n_vertices*i];
}

/** The class destructor frees the dynamically allocated memory for storing
 * neighbor information. */
voronoicell_neighbor::~voronoicell_neighbor() {
	for(int i=current_vertex_order-1;i>=0;i--) if(mem[i]>0) delete [] mne[i];
	delete [] mne;
	delete [] ne;
}

/** Computes a vector list of neighbors. */
void voronoicell_neighbor::neighbors(std::vector<int> &v) {
	v.clear();
	int i,j,k,l,m;
	for(i=1;i<p;i++) for(j=0;j<nu[i];j++) {
		k=ed[i][j];
		if(k>=0) {
			v.push_back(ne[i][j]);
			ed[i][j]=-1-k;
			l=cycle_up(ed[i][nu[i]+j],k);
			do {
				m=ed[k][l];
				ed[k][l]=-1-m;
				l=cycle_up(ed[k][nu[k]+l],m);
				k=m;
			} while (k!=i);
		}
	}
	reset_edges();
}

/** Prints the vertices, their edges, the relation table, and also notifies if
 * any memory errors are visible. */
void voronoicell_base::print_edges() {
	int j;
	double *ptsp=pts;
	for(int i=0;i<p;i++,ptsp+=4) {
		printf("%d %d  ",i,nu[i]);
		for(j=0;j<nu[i];j++) printf(" %d",ed[i][j]);
		printf("  ");
		while(j<(nu[i]<<1)) printf(" %d",ed[i][j]);
		printf("   %d",ed[i][j]);
		print_edges_neighbors(i);
		printf("  %g %g %g %p",*ptsp,ptsp[1],ptsp[2],(void*) ed[i]);
		if(ed[i]>=mep[nu[i]]+mec[nu[i]]*((nu[i]<<1)+1)) puts(" Memory error");
		else puts("");
	}
}

/** This prints out the neighbor information for vertex i. */
void voronoicell_neighbor::print_edges_neighbors(int i) {
	if(nu[i]>0) {
		int j=0;
		printf("     (");
		while(j<nu[i]-1) printf("%d,",ne[i][j++]);
		printf("%d)",ne[i][j]);
	} else printf("     ()");
}

// Explicit instantiation
template bool voronoicell_base::nplane(voronoicell&,double,double,double,double,int);
template bool voronoicell_base::nplane(voronoicell_neighbor&,double,double,double,double,int);
template void voronoicell_base::check_memory_for_copy(voronoicell&,voronoicell_base*);
template void voronoicell_base::check_memory_for_copy(voronoicell_neighbor&,voronoicell_base*);

}
