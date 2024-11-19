#pragma once

#include <vector>
#include <memory>
#include <format>

#include <SFML/Graphics.hpp>

#include "compute.hpp"

template <typename T, int N>
class Interface {
	std::shared_ptr<FractalComputer<T, N>> computer;

	sf::RenderWindow window;
	sf::Texture texture;
	sf::Sprite sprite;
	sf::Font infoFont;
	std::vector<sf::Text> infoTexts;
	sf::RectangleShape infoRect;

	std::vector<unsigned char> pix;
	std::vector<sf::Color> color_map;

	bool showInfos;

    public:
	Interface(std::shared_ptr<FractalComputer<T, N>> computer_, std::size_t fpsLimit = 60,
		  std::vector<sf::Color> const& cmap = { { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 } })
		: computer{ computer_ },
		  window{ sf::VideoMode(computer->getWidth(), computer->getHeight()), "Newton fractal viewer" },
		  infoTexts{ 5 }, pix(computer->getWidth() * computer->getHeight() * 4), color_map{ cmap },
		  showInfos{ false } {
		window.setFramerateLimit(fpsLimit);
		texture.create(computer->getWidth(), computer->getHeight());
		sprite = sf::Sprite{ texture };
		if (!infoFont.loadFromFile("res/arial.ttf")) {
			throw std::runtime_error("Could not load font!");
		}
		infoRect.setFillColor({ 128, 128, 128, 150 }); // grey
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
			/*} else if (event.type == sf::Event::Resized) {
			computer->updateWidth(event.size.width);
			computer->updateHeight(event.size.height);*/ // buffer are not easily resizable
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
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::RControl) ||
				    sf::Keyboard::isKeyPressed(sf::Keyboard::LControl)) {
					computer->increaseIters(1);
				} else {
					computer->zoomIn(1);
				}
			} else if (event.key.code == sf::Keyboard::Subtract || event.key.code == sf::Keyboard::Dash) {
				if (sf::Keyboard::isKeyPressed(sf::Keyboard::RControl) ||
				    sf::Keyboard::isKeyPressed(sf::Keyboard::LControl)) {
					computer->decreaseIters(1);
				} else {
					computer->zoomOut(1);
				}
			} else if (event.key.code == sf::Keyboard::I) {
				toggleInformations();
			}
		}
	}

	std::array<std::string, 5> infoStrings() const {
		std::array<std::string, 5> ret;
		ret[0] = std::format("FLOPS: {:.2e}", computer->getFLOPS());
		ret[1] = std::format("s/it: {:.2f}", computer->getIterTime());
		ret[2] = std::format("center: ({:.4f}{:+.4f}i)", computer->getCenter().re, computer->getCenter().im);
		ret[3] = std::format("resolution: {:.2e}", computer->getIncrement());
		ret[4] = std::format("cycles: {:d}", computer->getCycles());
		return ret;
	}

	void drawInfos(float spacing = 10.f) {
		if (!showInfos)
			return;

		infoTexts.clear();
		for (auto const& is : infoStrings()) {
			sf::Text text(is, infoFont, 20);
			text.setFillColor(sf::Color::White);
			infoTexts.push_back(std::move(text));
		}

		auto wsize = window.getSize();
		infoRect.setSize(sf::Vector2f(wsize.x * 0.4, wsize.y * 0.4));
		infoRect.setPosition(wsize.x * 0.1, wsize.y * 0.1);
		window.draw(infoRect);

		float curY = infoRect.getPosition().y + spacing;
		for (auto& text : infoTexts) {
			text.setPosition(infoRect.getPosition().x + spacing, curY);
			curY += text.getLocalBounds().height + spacing;
			window.draw(text);
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
			drawInfos();
			window.display();
		}
	}
};
