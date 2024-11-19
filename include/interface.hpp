#pragma once

#include <memory>

#include <SFML/Graphics.hpp>

#include "compute.hpp"

template <typename T, int N>
class Interface {
	std::shared_ptr<FractalComputer<T, N>> computer;

	sf::RenderWindow window;
	sf::Texture texture;
	sf::Sprite sprite;

	std::vector<unsigned char> pix;
	std::vector<sf::Color> color_map;

	bool showInfos;

    public:
	Interface(std::shared_ptr<FractalComputer<T, N>> computer_,
		  std::vector<sf::Color> const& cmap = { { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 } },
		  std::size_t fpsLimit = 60)
		: computer{ computer_ },
		  window{ sf::VideoMode(computer->getWidth(), computer->getHeight()), "Newton fractal viewer" },
		  pix(computer->getWidth() * computer->getHeight() * 4), color_map{ cmap }, showInfos{ false } {
		window.setFramerateLimit(fpsLimit);
		texture.create(computer->getWidth(), computer->getHeight());
		sprite = sf::Sprite{ texture };
	}

	std::weak_ptr<FractalComputer<T, N>> getComputer() const { return computer; }

	void updateSprite() {
		auto const& outVec = computer->compute();
		for (std::size_t i = 0; i < outVec.size(); ++i) {
			auto idx = i * 4;
			auto const r = outVec[i];
			auto const& col = color_map[r];
			pix[idx + 0] = col.r;
			pix[idx + 1] = col.g;
			pix[idx + 2] = col.b;
			pix[idx + 3] = col.a;
		}
		texture.update(pix.data());
	}

	void toggleInformations() { showInfos = !showInfos; }

	void handleEvent(sf::Event const& event) {
		if (event.type == sf::Event::Closed) {
			window.close();
		} else if (event.type == sf::Event::KeyPressed) {
			if (event.key.code == sf::Keyboard::Escape) {
				window.close();
			} else if (event.key.code == sf::Keyboard::Left) {
				computer->moveLeft(10);
			} else if (event.key.code == sf::Keyboard::Right) {
				computer->moveRight(10);
			} else if (event.key.code == sf::Keyboard::Up) {
				computer->moveUp(10);
			} else if (event.key.code == sf::Keyboard::Down) {
				computer->moveDown(10);
			} else if (event.key.code == sf::Keyboard::Add || event.key.code == sf::Keyboard::Equal) {
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::RShift) ||
				    sf::Keyboard::isKeyPressed(sf::Keyboard::LShift)) {
					computer->increaseIters(1);
				} else {
					computer->zoomIn(1);
				}
			} else if (event.key.code == sf::Keyboard::Subtract || event.key.code == sf::Keyboard::Dash) {
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::RShift) ||
				    sf::Keyboard::isKeyPressed(sf::Keyboard::LShift)) {
					computer->decreaseIters(1);
				} else {
					computer->zoomOut(1);
				}
			} else if (event.key.code == sf::Keyboard::I) {
				toggleInformations();
			}
		}
	}

	void play() {
		while (window.isOpen()) {
			sf::Event event;
			while (window.pollEvent(event)) {
				handleEvent(event);
			}

			updateSprite();
			window.clear();
			window.draw(sprite);
			window.display();
		}
	}
};
