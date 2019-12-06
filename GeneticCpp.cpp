// Tested with Visual Studio 2019 - version 16.4.0
// Set to Release - x64 when work with large number
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
using namespace std;

// constants

static const double LOCATION_RANGE = 1000.0; // Map size in km
static const int POPULATION_SIZE = 1000;
static const int DNA_SIZE = 30; // Number of cities, a city is defined by its id and its location (x, y)
static const int GENERATION_COUNT = 10000; // Number of evolution loops
static const double MUTATION_CHANCE = 0.05;
static const double SELECTION_SIZE = 0.6; // Select 60% top individuals each loop
static const double CROSSOVER_POINT = 0.5; // Child inherit 50% of father genes and 50% of mother genes

// class prototypes

class Gene {
public:
	Gene(const int id, const double x, const double y);
	const double calcDistance(const Gene& city) const;
	const int getId() const { return id; }
	const string toString() const;
private:
	int id;
	double x;
	double y;
};

class Chromosome {
public:
	Chromosome(const vector<Gene>& cities);
	const string toString() const;
	const vector<Gene>& getDNA() const;
	const double getFitness() const;
private:
	const void evaluateFitness();
private:
	vector<Gene> DNA;
	double fitness;
};

// global variables

ofstream out("output.txt");

random_device rd {};
mt19937 gen {rd()};
uniform_real_distribution<double> distributionLocation {-LOCATION_RANGE, LOCATION_RANGE};
uniform_real_distribution<double> distributionPercentage {0.0, 1.0};

vector<Gene> originalDNA;
vector<Chromosome> population;

int mutationCount;
double avgFitness;

// function prototypes

void initPopulation(const int chromosomesCount, const int genesCount = 0);
void printDNA(const Chromosome& individual);
void populationSort();
void selection();
void printIndividualWithBestFitness(const int generation);
void crossover();
void mutation();
void calcAvgFitness();

// main function

int main() {
	auto started = chrono::high_resolution_clock::now();

	initPopulation(POPULATION_SIZE, DNA_SIZE);
	out << "Input (original individual):" << endl;
	printDNA(originalDNA);
	for (int i = 0; i < GENERATION_COUNT; ++i) {
		calcAvgFitness();
		selection();
		printIndividualWithBestFitness(i);
		crossover();
		mutation();
		/*out << i << "th - size: " << population.size() << " individuals:" << endl;
		for (auto individual : population) {
			out << individual.toString() << endl;
		}*/
	}
	populationSort();
	printIndividualWithBestFitness(GENERATION_COUNT);
	out << "Mutation occurs " << mutationCount << " times" << endl;

	auto done = chrono::high_resolution_clock::now();

	out << "Done!\nExecution time: " << chrono::duration_cast<chrono::seconds>(done - started).count() << "s" << endl;
	cout << "Done!\nExecution time: " << chrono::duration_cast<chrono::seconds>(done - started).count() << "s" << endl;
	
	out.close();
	return 0;
}

// Gene implementation

Gene::Gene(const int id, const double x, const double y) {
	this->id = id;
	this->x = x;
	this->y = y;
}

const double Gene::calcDistance(const Gene& city) const {
	return sqrt((this->x - city.x) * (this->x - city.x) + (this->y - city.y) * (this->y - city.y));
}

const string Gene::toString() const {
	return "{" + to_string(this->id) + ": (" + to_string(this->x) + ", " + to_string(this->y) + ")}";
}

// Chromosome implementation

Chromosome::Chromosome(const vector<Gene>& cities) {
	this->DNA = cities;
	evaluateFitness();
}

const void Chromosome::evaluateFitness() {
	auto fitness = 0.0;
	for (int i = 1; i < this->DNA.size(); ++i) {
		fitness += this->DNA[i].calcDistance(this->DNA[i - 1]);
	}
	fitness += this->DNA[this->DNA.size() - 1].calcDistance(this->DNA[0]);
	this->fitness = LOCATION_RANGE * LOCATION_RANGE / fitness;
}

const string Chromosome::toString() const {
	string output = "[";
	for (int i = 0; i < this->DNA.size() - 1; ++i) {
		output += to_string(this->DNA[i].getId()) + ", ";
	}
	output += to_string(this->DNA[this->DNA.size() - 1].getId()) + "] -> " + to_string(this->fitness);
	return output;
}

const vector<Gene>& Chromosome::getDNA() const {
	return this->DNA;
}

const double Chromosome::getFitness() const {
	return this->fitness;
}

// functions implementation

void initPopulation(const int chromosomesCount, const int genesCount) {
	if (genesCount != 0) {
		for (int i = 0; i < genesCount; ++i) {
			int id = i;
			double x = distributionLocation(gen);
			double y = distributionLocation(gen);
			originalDNA.emplace_back(id, x, y);
		}
		population.push_back(originalDNA);
		for (int i = 1; i < chromosomesCount; ++i) {
			auto tempDNA = originalDNA;
			random_shuffle(tempDNA.begin(), tempDNA.end());
			population.push_back(tempDNA);
		}
	} else {
		for (int i = 0; i < chromosomesCount; ++i) {
			auto tempDNA = originalDNA;
			random_shuffle(tempDNA.begin(), tempDNA.end());
			population.push_back(tempDNA);
		}
	}
}

void printDNA(const Chromosome& individual) {
	out << individual.toString() << endl;
	for (auto gene : individual.getDNA()) {
		out << gene.toString() << endl;
	}
}

void populationSort() {
	stable_sort(population.begin(), population.end(),
		[](const Chromosome& a, const Chromosome& b) -> bool {
			return a.getFitness() > b.getFitness();
		});
}

void selection() {
	populationSort();
	int selectionSize = (int)(POPULATION_SIZE * SELECTION_SIZE);
	population.erase(population.begin() + selectionSize, population.end());
	initPopulation(POPULATION_SIZE - selectionSize / 2 * 3);
}

void printIndividualWithBestFitness(const int generation) {
	out << "\nGeneration " << generation << " with " << POPULATION_SIZE << " individuals. Average fitness: " << avgFitness << endl;
	out << "-> Current best fitness:";
	out << population[0].toString() << endl;
}

void crossover() {
	random_shuffle(population.begin(), population.end());
	int selectionSize = (int)(POPULATION_SIZE * SELECTION_SIZE);
	for (int i = 0; i < selectionSize / 2; ++i) {
		auto father = population[i].getDNA();
		auto mother = population[population.size() - 1 - i].getDNA();
		vector<Gene> child{father.begin(), father.begin() + (int)(father.size() * CROSSOVER_POINT)};
		for (auto geneOfMother : mother) {
			if (find_if(child.begin(), child.end(),
				[&](const Gene& gene) -> bool {
					return gene.getId() == geneOfMother.getId();
				}) != child.end()) {
				continue;
			}
			child.push_back(geneOfMother);
		}
		population.push_back(child);
	}
}

void mutation() {
	uniform_int_distribution<int> distributionDNA{0, (int)population.size()};
	double chance = distributionPercentage(gen);
	if (chance <= MUTATION_CHANCE) {
		swap(population[distributionDNA(gen)], population[distributionDNA(gen)]);
		++mutationCount;
	}
}

void calcAvgFitness() {
	avgFitness = 0.0;
	for (auto individual : population) {
		avgFitness += individual.getFitness();
	}
	avgFitness /= population.size();
}
