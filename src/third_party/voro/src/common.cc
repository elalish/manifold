// Voro++, a 3D cell-based Voronoi library
//
// Author   : Chris H. Rycroft (Harvard University / LBL)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

/** \file common.cc
 * \brief Implementations of the small helper functions. */

#include "common.hh"

namespace voro {

void check_duplicate(int n,double x,double y,double z,int id,double *qp) {
	double dx=*qp-x,dy=qp[1]-y,dz=qp[2]-z;
	if(dx*dx+dy*dy+dz*dz<1e-10) {
		printf("Duplicate: %d (%g,%g,%g) matches %d (%g,%g,%g)\n",n,x,y,z,id,*qp,qp[1],qp[2]);
		exit(1);
	}
}

/** \brief Function for printing fatal error messages and exiting.
 *
 * Function for printing fatal error messages and exiting.
 * \param[in] p a pointer to the message to print.
 * \param[in] status the status code to return with. */
void voro_fatal_error(const char *p,int status) {
	fprintf(stderr,"voro++: %s\n",p);
	exit(status);
}

/** \brief Prints a vector of positions.
 *
 * Prints a vector of positions as bracketed triplets.
 * \param[in] v the vector to print.
 * \param[in] fp the file stream to print to. */
void voro_print_positions(std::vector<double> &v,FILE *fp) {
	if(v.size()>0) {
		fprintf(fp,"(%g,%g,%g)",v[0],v[1],v[2]);
		for(int k=3;(unsigned int) k<v.size();k+=3) {
			fprintf(fp," (%g,%g,%g)",v[k],v[k+1],v[k+2]);
		}
	}
}

/** \brief Opens a file and checks the operation was successful.
 *
 * Opens a file, and checks the return value to ensure that the operation
 * was successful.
 * \param[in] filename the file to open.
 * \param[in] mode the cstdio fopen mode to use.
 * \return The file handle. */
FILE* safe_fopen(const char *filename,const char *mode) {
	FILE *fp=fopen(filename,mode);
	if(fp==NULL) {
		fprintf(stderr,"voro++: Unable to open file '%s'\n",filename);
		exit(VOROPP_FILE_ERROR);
	}
	return fp;
}

/** \brief Prints a vector of integers.
 *
 * Prints a vector of integers.
 * \param[in] v the vector to print.
 * \param[in] fp the file stream to print to. */
void voro_print_vector(std::vector<int> &v,FILE *fp) {
	int k=0,s=v.size();
	while(k+4<s) {
		fprintf(fp,"%d %d %d %d ",v[k],v[k+1],v[k+2],v[k+3]);
		k+=4;
	}
	if(k+3<=s) {
		if(k+4==s) fprintf(fp,"%d %d %d %d",v[k],v[k+1],v[k+2],v[k+3]);
		else fprintf(fp,"%d %d %d",v[k],v[k+1],v[k+2]);
	} else {
		if(k+2==s) fprintf(fp,"%d %d",v[k],v[k+1]);
		else fprintf(fp,"%d",v[k]);
	}
}

/** \brief Prints a vector of doubles.
 *
 * Prints a vector of doubles.
 * \param[in] v the vector to print.
 * \param[in] fp the file stream to print to. */
void voro_print_vector(std::vector<double> &v,FILE *fp) {
	int k=0,s=v.size();
	while(k+4<s) {
		fprintf(fp,"%g %g %g %g ",v[k],v[k+1],v[k+2],v[k+3]);
		k+=4;
	}
	if(k+3<=s) {
		if(k+4==s) fprintf(fp,"%g %g %g %g",v[k],v[k+1],v[k+2],v[k+3]);
		else fprintf(fp,"%g %g %g",v[k],v[k+1],v[k+2]);
	} else {
		if(k+2==s) fprintf(fp,"%g %g",v[k],v[k+1]);
		else fprintf(fp,"%g",v[k]);
	}
}

/** \brief Prints a vector a face vertex information.
 *
 * Prints a vector of face vertex information. A value is read, which
 * corresponds to the number of vertices in the next face. The routine reads
 * this number of values and prints them as a bracked list. This is repeated
 * until the end of the vector is reached.
 * \param[in] v the vector to interpret and print.
 * \param[in] fp the file stream to print to. */
void voro_print_face_vertices(std::vector<int> &v,FILE *fp) {
	int j,k=0,l;
	if(v.size()>0) {
		l=v[k++];
		if(l<=1) {
			if(l==1) fprintf(fp,"(%d)",v[k++]);
			else fputs("()",fp);
		} else {
			j=k+l;
			fprintf(fp,"(%d",v[k++]);
			while(k<j) fprintf(fp,",%d",v[k++]);
			fputs(")",fp);
		}
		while((unsigned int) k<v.size()) {
			l=v[k++];
			if(l<=1) {
				if(l==1) fprintf(fp," (%d)",v[k++]);
				else fputs(" ()",fp);
			} else {
				j=k+l;
				fprintf(fp," (%d",v[k++]);
				while(k<j) fprintf(fp,",%d",v[k++]);
				fputs(")",fp);
			}
		}
	}
}

}
